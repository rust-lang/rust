trait Bar {
    const X: usize;
    fn return_n(&self) -> [u8; Bar::X]; //~ ERROR: E0790
}

impl dyn Bar {} //~ ERROR: the trait `Bar` is not dyn compatible

fn main() {}
