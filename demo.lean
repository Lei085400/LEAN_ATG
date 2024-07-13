import Lean4Repl
import Mathlib.Data.Nat.Choose.Sum
import Mathlib.Data.Real.Basic
import Mathlib.Data.Finset.LocallyFinite


#align_import data.nat.choose.sum from "leanprover-community/mathlib"@"4c19a16e4b705bf135cf9a80ac18fcc99c438514"

open Nat
open Finset

open BigOperators

set_option maxHeartbeats 999999999999999999999999

-- theorem idt1_Pascal's_Recurrence(h1:1 ≤ n)(h2:1 ≤ k) : choose n k = choose (n-1) k  + choose (n-1) (k-1) := by
--   have choose_eq_choose_sub_add :  choose n k = choose (n - 1 + 1) (k - 1 + 1)  := by
--    rw[Nat.sub_add_cancel h1, Nat.sub_add_cancel h2]
--   rw[choose_eq_choose_sub_add]
--   rw[add_comm (choose (n - 1) k) (choose (n - 1) (k - 1))]
--   have choose_sub_eq_choose_sub_add : choose (n - 1) k = choose (n - 1) (k - 1 + 1) := by rw[Nat.sub_add_cancel h2]
--   rw[choose_sub_eq_choose_sub_add, choose_succ_succ]

-- theorem idt1_Pascal's_Recurrence(h1:1 ≤ n)(h2:1 ≤ k) : choose n k = choose (n-1) k  + choose (n-1) (k-1) := by lean_repl sorry
--  have choose_eq_choose_sub_add :  choose n k = choose (n - 1 + 1) (k - 1 + 1)  := by rw[Nat.sub_add_cancel h1, Nat.sub_add_cancel h2]
--   rw[add_comm]
--   simp [*, choose]
--   rw[Nat.sub_add_cancel h2]

-- theorem choose_symm {n k : ℕ} (hk : k ≤ n) : choose n (n - k) = choose n k := by
--   rw [choose_eq_factorial_div_factorial hk]
--   rw [choose_eq_factorial_div_factorial]
--   rw [Nat.sub_sub_self hk]
--   rw [mul_comm]
--   exact Nat.sub_le _ _

-- theorem choose_symm' {n k : ℕ} (hk : k ≤ n) : choose n (n - k) = choose n k := by lean_repl sorry
--   rw [choose_eq_factorial_div_factorial hk, choose_eq_factorial_div_factorial (Nat.sub_le _ _),
--     tsub_tsub_cancel_of_le hk, mul_comm]



-- theorem choose_eq_choose_sub_add(h1:1 ≤ n)(h2:1 ≤ k) :  choose n k = choose (n - 1 + 1) (k - 1 + 1)  := by lean_repl sorry
--     rw[Nat.sub_add_cancel h1, Nat.sub_add_cancel h2]

-- theorem sum_neg_pow_mul_mul (t : ℕ → ℕ → ℕ → ℝ := fun m n k => (-1 : ℝ)^k * (choose n k : ℝ) * (m / (m+k))):
--     ∑ k in Ico 1 n, (-1 : ℝ)^k * ((choose (n-1) k + choose (n-1) (k-1)) : ℝ) * (m / (m+k)) = ∑ k in Ico 1 n, (((-1 : ℝ)^k * choose (n-1) k * (m / (m+k)) + (-1 : ℝ)^k * choose (n-1) (k-1) * (m / (m+k))) : ℝ):= by lean_repl sorry
--     -- congr 1
    -- ext1 k
    -- rw [mul_add]
    -- rw [add_mul]
  -- refine' sum_congr rfl fun y _ => _
  -- rw[mul_add]
  -- rw[add_mul]
--   -- apply sum_congr rfl
--   -- intro x hx
--   -- congr 2
--   -- rw [mul_add]
--   -- rw [add_mul]

-- theorem lt_eq_le_sub{y:ℕ}(hn0 : 0 < n) : y < n → y ≤ n - 1 := by lean_repl sorry
--     rw[Nat.lt_iff_le_pred hn0]
--     intro a
--     exact a

-- theorem itd53 (hn :0 < n) :  ∑ k in range (n+1), (choose n k : ℝ) * (-1 : ℝ) ^ k / (k + 2) = 1 / ((n + 1) * (n + 2)) := by lean_repl sorry

--theorem itd53_ (hn :0 < n) :  ∑ k in range (n+1), (choose n k : ℝ) * (((-1 : ℝ) ^ (k+2)) / (k + 2)) = -(1 / (n + 2)) + 1 / (n+1) := by lean_repl sorry

-- theorem l₄ : 1/(k+1) - 1/(k+2) = 1/((k+1)*(k+2)) := by lean_repl sorry

-- theorem lt_eq_le_sub{y:ℕ}(hn0 : 0 < n) : y < n → y ≤ n - 1 := by lean_repl sorry

-- theorem Real_choose_eq_choose_add(h1:1 ≤ n)(h2:1 ≤ k) : (choose (n - 1 + 1) (k - 1 + 1) : ℝ) = (choose (n - 1 + 1 - 1) (k - 1 + 1) : ℝ)  + (choose (n - 1 + 1 - 1) (k - 1 + 1 - 1): ℝ) := by lean_repl sorry
--   -- rw[← Nat.sub_add_cancel h1, ← Nat.sub_add_cancel h2]
--   simp
--   rw[choose_succ_succ]
--   rw[add_comm]
--   simp

example (a b c : Nat) : a + b + c = a + c + b := by
  rw [Nat.add_assoc]
  rw [Nat.add_comm b c]
  rw [Nat.add_assoc]

theorem abc (a b c : Nat) : a + b + c = a + c + b := by
  rw [Nat.add_comm]
  rw [Nat.add_assoc]


example(a b c : Nat): c + (a + b) = a + (c + b) := by
  have habc : a + b + c = a + c + b := by rw[abc]
  rw[Nat.add_comm] at habc
  rw[Nat.add_assoc] at habc
  rw[habc]


example(a b c:Nat):b + (c + a) = a + (c + b):=by
  rw [← Nat.add_assoc]
  rw [Nat.add_comm b c]
  rw [Nat.add_comm a (c+b)]


theorem sub_add_cancel''' {n m : Nat} (h : m ≤ n) : n - m + m = n := by
  rw [Nat.add_comm]
  rw [Nat.add_sub_of_le h]
