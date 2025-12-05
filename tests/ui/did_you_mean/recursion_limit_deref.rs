//~ ERROR reached the recursion limit finding the struct tail for `K`
//~| ERROR reached the recursion limit finding the struct tail for `Bottom`

// Test that the recursion limit can be changed and that the compiler
// suggests a fix. In this case, we have a long chain of Deref impls
// which will cause an overflow during the autoderef loop.
//@ compile-flags: -Zdeduplicate-diagnostics=yes

#![allow(dead_code)]
#![recursion_limit="10"]

macro_rules! link {
    ($outer:ident, $inner:ident) => {
        struct $outer($inner);
        //~^ ERROR reached the recursion limit finding the struct tail for `Bottom`

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
    }
}

struct Bottom;
impl Bottom {
    fn new() -> Bottom {
        Bottom
    }
}

link!(Top, A);
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

fn main() {
    let t = Top::new();
    let x: &Bottom = &t; //~ ERROR mismatched types
    //~^ error recursion limit
}
