// build-fail
// error-pattern: reached the type-length limit while instantiating
// compile-flags: -Copt-level=0
// normalize-stderr-test: ".nll/" -> "/"

// Test that the type length limit can be changed.
// The exact type depends on optimizations, so disable them.

#![allow(dead_code)]
#![type_length_limit="4"]

macro_rules! link {
    ($id:ident, $t:ty) => {
        pub type $id = ($t, $t, $t);
    }
}

link! { A, B }
link! { B, C }
link! { C, D }
link! { D, E }
link! { E, F }
link! { F, G }

pub struct G;

fn main() {
    drop::<Option<A>>(None);
}
