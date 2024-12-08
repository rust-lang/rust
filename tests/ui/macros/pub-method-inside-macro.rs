//@ run-pass
// Issue #17436


mod bleh {
    macro_rules! foo {
        () => {
            pub fn bar(&self) { }
        }
    }

    pub struct S;

    impl S {
        foo!();
    }
}

fn main() {
    bleh::S.bar();
}
