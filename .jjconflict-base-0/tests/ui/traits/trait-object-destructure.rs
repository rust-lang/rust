//! Regression test for destructuring trait references (`&dyn T`).
//! Checks cases where number of `&` patterns (n) matches/doesn't match references (m).
//!
//! Issue: https://github.com/rust-lang/rust/issues/15031

trait T {
    fn foo(&self) {}
}

impl T for isize {}

fn main() {
    // Valid cases: n < m (can dereference)
    let &x = &(&1isize as &dyn T);
    let &x = &&(&1isize as &dyn T);
    let &&x = &&(&1isize as &dyn T);

    // Error cases: n == m (cannot dereference trait object)
    let &x = &1isize as &dyn T; //~ ERROR type `&dyn T` cannot be dereferenced
    let &&x = &(&1isize as &dyn T); //~ ERROR type `&dyn T` cannot be dereferenced

    // Error cases: n > m (type mismatch)
    let &&x = &1isize as &dyn T; //~ ERROR mismatched types
    let &&&x = &(&1isize as &dyn T); //~ ERROR mismatched types
}
