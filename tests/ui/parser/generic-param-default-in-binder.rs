// Check that defaults for generic parameters in `for<...>` binders are
// syntactically valid. See also PR #119042.

//@ check-pass

macro_rules! a { ($ty:ty) => {} }

a! { for<T = &i32> fn() }

fn main() {}
