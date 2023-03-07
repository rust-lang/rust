// run-pass
// Issue #17436

// pretty-expanded FIXME #23616

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
