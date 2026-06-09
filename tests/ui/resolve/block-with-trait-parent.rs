//@ check-pass

trait Trait {
    fn method(&self) {
        // Items inside a block turn it into a module internally.
        struct S;
        impl Trait for S {}

        // OK, `Trait` is in scope here from method resolution point of view.
        S.method();
    }
}

fn main() {}
