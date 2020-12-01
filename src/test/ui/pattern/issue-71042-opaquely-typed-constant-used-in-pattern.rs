#![feature(impl_trait_in_bindings)]
#![allow(incomplete_features)]

fn main() {
    const C: impl Copy = 0;
    match C {
        C | //~ ERROR: `impl Copy` cannot be used in patterns
        _ => {}
    }
}
