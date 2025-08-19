//! Regression test for destructuring trait references (`&dyn T`/`Box<dyn T>`).
//! Checks cases where number of `&`/`Box` patterns (n) matches/doesn't match references (m).
//!
//! Issue: https://github.com/rust-lang/rust/issues/15031

#![feature(box_patterns)]

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
    let box x = Box::new(1isize) as Box<dyn T>; //~ ERROR type `Box<dyn T>` cannot be dereferenced

    // Error cases: n > m (type mismatch)
    let &&x = &1isize as &dyn T; //~ ERROR mismatched types
    let &&&x = &(&1isize as &dyn T); //~ ERROR mismatched types
    let box box x = Box::new(1isize) as Box<dyn T>; //~ ERROR mismatched types
}
