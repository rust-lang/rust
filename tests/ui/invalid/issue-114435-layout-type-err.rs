//@ check-fail
//@ compile-flags: --crate-type lib -Cdebuginfo=2

#![recursion_limit = "10"]
macro_rules! link {
    ($outer:ident, $inner:ident) => {
        struct $outer($inner);
        impl $outer {
            fn new() -> $outer {
                $outer($inner::new())
            }
        }
        impl std::ops::Deref for $outer {
            type Target = $inner;
            fn deref(&self) -> &$inner {
                &self.0
            }
        }
    };
}

struct Bottom;

impl Bottom {
    fn new() -> Bottom {
        Bottom
    }
}

link!(A, B);
link!(B, C);
link!(C, D);
link!(D, E);
link!(E, F);
link!(F, G);
link!(G, H);
link!(H, I);
link!(I, J);
link!(J, K);
link!(K, Bottom);

fn main() {}

//~? ERROR reached the recursion limit finding the struct tail for `Bottom`
