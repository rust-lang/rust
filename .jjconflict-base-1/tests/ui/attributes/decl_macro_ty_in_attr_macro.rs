// tests for #137662: using a ty or (or most other) fragment inside an attr macro wouldn't work
// because of a missing code path. With $repr: tt it did work.
//@ check-pass

macro_rules! foo {
    {
        $repr:ty
    } => {
        #[repr($repr)]
        pub enum Foo {
            Bar = 0i32,
        }
    }
}

foo! {
    i32
}

fn main() {}
