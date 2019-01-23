pub trait FooMap {
    fn map<B, F: Fn() -> B>(&self, f: F) -> B;
}

impl FooMap for bool {
    fn map<B, F: Fn() -> B>(&self, f: F) -> B {
        f()
    }
}

fn main() {
    let a = true;
    a.map(|| false);
}
