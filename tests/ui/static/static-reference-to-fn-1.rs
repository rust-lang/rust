struct A<'a> {
    func: &'a fn() -> Option<isize>
}

impl<'a> A<'a> {
    fn call(&self) -> Option<isize> {
        (*self.func)()
    }
}

fn foo() -> Option<isize> {
    None
}

fn create() -> A<'static> {
    A {
        func: &foo, //~ ERROR mismatched types
    }
}

fn main() {
    let a = create();
    a.call();
}
