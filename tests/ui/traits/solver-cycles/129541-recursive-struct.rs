//~ ERROR reached the recursion limit finding the struct tail for `<[Hello] as Normalize>::Assoc`
// Regression test for #129541

//@ revisions: unique_curr unique_next multiple_curr multiple_next
//@ ignore-compare-mode-next-solver (explicit revisions)
//@[unique_next] compile-flags: -Znext-solver
//@[multiple_next] compile-flags: -Znext-solver

trait Bound {}
trait Normalize {
    type Assoc;
}

#[cfg(any(multiple_curr, multiple_next))]
impl<T: Bound> Normalize for T {
    type Assoc = T;
}
impl<T: Bound> Normalize for [T] {
    type Assoc = T;
}

impl Bound for Hello {}
struct Hello {
    a: <[Hello] as Normalize>::Assoc,
}

fn main() {}
