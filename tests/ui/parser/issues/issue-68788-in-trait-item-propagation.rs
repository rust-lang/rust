// Make sure we don't propagate restrictions on trait impl items to items inside them.

// check-pass
// edition:2018

fn main() {}

trait X {
    fn foo();
}

impl X for () {
    fn foo() {
        struct S;
        impl S {
            pub const X: u8 = 0;
            pub const fn bar() {}
            async fn qux() {}
        }
    }
}
