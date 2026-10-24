fn call<C>(f: C)
where
    C: Fn(),
{
    f()
}

struct FooPrinter {}

impl FooPrinter {
    fn print(&self) {
        println!("foo");
    }
    fn new() -> Self {
        FooPrinter {}
    }
}

fn main() {
    call(FooPrinter::new().print); //~ ERROR attempted to take value of method
}
