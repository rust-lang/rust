struct X(());
impl X {
    pub unsafe fn with(&self) { }
}

fn main() {
    X(()).with(); //~ ERROR requires unsafe function or block
}
