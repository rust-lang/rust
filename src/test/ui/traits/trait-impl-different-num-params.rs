trait foo {
    fn bar(&self, x: usize) -> Self;
}
impl foo for isize {
    fn bar(&self) -> isize {
        //~^ ERROR method `bar` has 1 parameter but the declaration in trait `foo::bar` has 2
        *self
    }
}

fn main() {
}
