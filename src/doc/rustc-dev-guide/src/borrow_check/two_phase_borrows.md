# Two-phase borrows

Two-phase borrows are a more permissive version of mutable borrows that allow
nested method calls such as `vec.push(vec.len())`. Such borrows first act as
shared borrows in a "reservation" phase and can later be "activated" into a
full mutable borrow.

Only certain implicit mutable borrows can be two-phase, any `&mut` or `ref mut`
in the source code is never a two-phase borrow. The cases where we generate a
two-phase borrow are:

1. The autoref borrow when calling a method with a mutable reference receiver.
2. A mutable reborrow in function arguments.
3. The implicit mutable borrow in an overloaded compound assignment operator.

To give some examples:

```rust2018
// In the source code

// Case 1:
let mut v = Vec::new();
v.push(v.len());
let r = &mut Vec::new();
r.push(r.len());

// Case 2:
std::mem::replace(r, vec![1, r.len()]);

// Case 3:
let mut x = std::num::Wrapping(2);
x += x;
```

Expanding these enough to show the two-phase borrows:

```rust,ignore
// Case 1:
let mut v = Vec::new();
let temp1 = &two_phase v;
let temp2 = v.len();
Vec::push(temp1, temp2);
let r = &mut Vec::new();
let temp3 = &two_phase *r;
let temp4 = r.len();
Vec::push(temp3, temp4);

// Case 2:
let temp5 = &two_phase *r;
let temp6 = vec![1, r.len()];
std::mem::replace(temp5, temp6);

// Case 3:
let mut x = std::num::Wrapping(2);
let temp7 = &two_phase x;
let temp8 = x;
std::ops::AddAssign::add_assign(temp7, temp8);
```

Whether a borrow can be two-phase is tracked by a flag on the [`AutoBorrow`]
after type checking, which is then [converted] to a [`BorrowKind`] during MIR
construction.

Each two-phase borrow is assigned to a temporary that is only used once. As
such we can define:

* The point where the temporary is assigned to is called the *reservation*
  point of the two-phase borrow.
* The point where the temporary is used, which is effectively always a
  function call, is called the *activation* point.

The activation points are found using the [`GatherBorrows`] visitor. The
[`BorrowData`] then holds both the reservation and activation points for the
borrow.

[`AutoBorrow`]: https://doc.rust-lang.org/nightly/nightly-rustc/rustc_middle/ty/adjustment/enum.AutoBorrow.html
[converted]: https://doc.rust-lang.org/nightly/nightly-rustc/rustc_mir_build/thir/cx/expr/trait.ToBorrowKind.html#method.to_borrow_kind
[`BorrowKind`]: https://doc.rust-lang.org/nightly/nightly-rustc/rustc_middle/mir/enum.BorrowKind.html
[`GatherBorrows`]: https://doc.rust-lang.org/nightly/nightly-rustc/rustc_borrowck/borrow_set/struct.GatherBorrows.html
[`BorrowData`]: https://doc.rust-lang.org/nightly/nightly-rustc/rustc_borrowck/borrow_set/struct.BorrowData.html

## Checking two-phase borrows

Two-phase borrows are treated as if they were mutable borrows with the
following exceptions:

1. At every location in the MIR we [check] if any two-phase borrows are
   activated at this location. If a live two phase borrow is activated at a
   location, then we check that there are no borrows that conflict with the
   two-phase borrow.
2. At the reservation point we error if there are conflicting live *mutable*
   borrows. And lint if there are any conflicting shared borrows.
3. Between the reservation and the activation point, the two-phase borrow acts
   as a shared borrow. We determine (in [`is_active`]) if we're at such a point
   by using the [`Dominators`] for the MIR graph.
4. After the activation point, the two-phase borrow acts as a mutable borrow.

[check]: https://doc.rust-lang.org/nightly/nightly-rustc/rustc_borrowck/struct.MirBorrowckCtxt.html#method.check_activations
[`Dominators`]: https://doc.rust-lang.org/nightly/nightly-rustc/rustc_data_structures/graph/dominators/struct.Dominators.html
[`is_active`]: https://doc.rust-lang.org/nightly/nightly-rustc/rustc_borrowck/path_utils/fn.is_active.html
