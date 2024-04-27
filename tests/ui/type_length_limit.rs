//@ build-fail
//@ error-pattern: reached the type-length limit while instantiating
//@ compile-flags: -Copt-level=0
//@ normalize-stderr-test: ".nll/" -> "/"

// Test that the type length limit can be changed.
// The exact type depends on optimizations, so disable them.

#![allow(dead_code)]
#![type_length_limit="8"]

macro_rules! link {
    ($id:ident, $t:ty) => {
        pub type $id = ($t, $t, $t);
    }
}

link! { A1, B1 }
link! { B1, C1 }
link! { C1, D1 }
link! { D1, E1 }
link! { E1, A }
link! { A, B }
link! { B, C }
link! { C, D }
link! { D, E }
link! { E, F }
link! { F, G<Option<i32>, Option<i32>> }

pub struct G<T, K>(std::marker::PhantomData::<(T, K)>);

fn main() {
    drop::<Option<A>>(None);
}
