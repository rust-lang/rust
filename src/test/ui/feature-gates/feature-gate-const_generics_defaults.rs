#[cfg(FALSE)]
struct A<const N: usize = 3>;
//~^ ERROR default values for const parameters are experimental

#[cfg(FALSE)]
fn foo<const B: bool = false>() {}
//~^ ERROR default values for const parameters are experimental

fn main() {}
