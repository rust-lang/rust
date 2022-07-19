trait Bar {
    const X: usize;
    fn return_n(&self) -> [u8; Bar::X]; //~ ERROR: E0790
}

impl dyn Bar {} //~ ERROR: the trait `Bar` cannot be made into an object

fn main() {}
