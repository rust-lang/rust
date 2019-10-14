trait Bar {
    const X: usize;
    fn return_n(&self) -> [u8; Bar::X]; //~ ERROR: type annotations needed
}

impl dyn Bar {} //~ ERROR: the trait `Bar` cannot be made into an object

fn main() {}
