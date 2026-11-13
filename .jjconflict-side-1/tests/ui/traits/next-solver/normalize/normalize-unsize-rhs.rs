//@ compile-flags: -Znext-solver
//@ check-pass

trait A {}
trait B: A {}

impl A for usize {}
impl B for usize {}

trait Mirror {
    type Assoc: ?Sized;
}

impl<T: ?Sized> Mirror for T {
    type Assoc = T;
}

fn main() {
    let x = Box::new(1usize) as Box<<dyn B as Mirror>::Assoc>;
    let y = x as Box<<dyn A as Mirror>::Assoc>;
}
