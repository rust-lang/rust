// An item exported only through a public glob, while also glob-imported into the
// same module through a facade with restricted visibility. The restricted duplicate
// must not make `f` unreachable: its optimized MIR must still be encoded for
// downstream crates (it is `cross_crate_inlinable`).

mod inner {
    pub fn f() -> u32 {
        42
    }
}

mod facade {
    #[allow(unused_imports)]
    pub(crate) use super::inner::f;
}

#[allow(unused_imports)]
use facade::*;
pub use inner::*;
