// When an uncalled method is passed where a callable is expected, suggest wrapping the
// method call in a closure instead of calling it. The closure forwards as many arguments
// as the method itself takes.

fn call<C>(f: C)
where
    C: Fn(),
{
    f()
}

fn call_with<C: Fn(u32, u32)>(f: C) {
    f(1, 2)
}

struct Runner;

impl Runner {
    fn run<C: FnOnce()>(&self, f: C) {
        f()
    }
}

struct FooPrinter {}

impl FooPrinter {
    fn print(&self) {
        println!("foo");
    }
    fn print_sum(&self, a: u32, b: u32) {
        println!("{}", a + b);
    }
    fn new() -> Self {
        FooPrinter {}
    }
}

fn takes_value(_: u32) {}

fn main() {
    call(FooPrinter::new().print); //~ ERROR attempted to take value of method
    call_with(FooPrinter::new().print_sum); //~ ERROR attempted to take value of method
    Runner.run(FooPrinter::new().print); //~ ERROR attempted to take value of method
    call((FooPrinter::new().print)); //~ ERROR attempted to take value of method
    // No callable is expected here, so keep suggesting to call the method.
    takes_value(FooPrinter::new().print_sum); //~ ERROR attempted to take value of method
}
