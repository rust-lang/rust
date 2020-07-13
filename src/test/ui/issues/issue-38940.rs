// issue-38940: error printed twice for deref recursion limit exceeded
// Test that the recursion limit can be changed. In this case, we have
// deeply nested types that will fail the `Send` check by overflow
// when the recursion limit is set very low.
#![allow(dead_code)]
#![recursion_limit="10"]
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
    let x: &Bottom = &t;
    //~^ ERROR mismatched types
    //~| ERROR reached the recursion limit while auto-dereferencing `J`
}
