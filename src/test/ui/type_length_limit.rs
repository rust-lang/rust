// build-fail
// error-pattern: reached the type-length limit while instantiating
// normalize-stderr-test: ".nll/" -> "/"

// Test that the type length limit can be changed.

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
