struct x(());
impl x {
    pub unsafe fn with(&self) { }
}

fn main() {
    x(()).with(); //~ ERROR requires unsafe function or block
}
