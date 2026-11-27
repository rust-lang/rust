//! Regression test for <https://github.com/rust-lang/rust/issues/120922>
//@ check-fail

fn serialize<T, S>() -> Result<(), ()> {
    Ok(())
}

trait Serialize {
    fn serialize();
}

impl Serialize for () {
    fn serialize() {
        {
            struct SerializeWith;
            impl SerializeWith {
                fn serialize() -> Result<(), ()> {
                    serialize() //~ ERROR type annotations needed
                }
            }
        };
    }
}

fn main() {}
