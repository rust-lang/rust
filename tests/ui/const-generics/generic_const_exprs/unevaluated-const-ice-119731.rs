// rust-lang/rust#119731
// ICE ... unevaluated constant UnevaluatedConst

#![feature(generic_const_exprs)]
#![allow(incomplete_features)]

mod v20 {
    const v4: usize = 512;
    pub type v11 = [[usize; v4]; v4];
    //~^ WARN type `v11` should have an upper camel case name
    const v2: v11 = [[256; v4]; v4];

    const v0: [[usize; v4]; v4] = v6(v8);
    //~^ ERROR cannot find value `v8` in this scope
    //~| ERROR cannot find function `v6` in this scope
    pub struct v17<const v10: usize, const v7: v11> {
        //~^ WARN type `v17` should have an upper camel case name
        //~| ERROR `[[usize; v4]; v4]` is forbidden as the type of a const generic parameter
        _p: (),
    }

    impl v17<512, v0> {
        pub const fn v21() -> v18 {}
        //~^ ERROR cannot find type `v18` in this scope
        //~| ERROR duplicate definitions with name `v21`
    }

    impl<const v10: usize> v17<v10, v2> {
        //~^ ERROR maximum number of nodes exceeded in constant v20::v17::<v10, v2>::{constant#0}
        //~| ERROR maximum number of nodes exceeded in constant v20::v17::<v10, v2>::{constant#0}
        //~| ERROR maximum number of nodes exceeded in constant v20::v17::<v10, v2>::{constant#0}
        //~| ERROR maximum number of nodes exceeded in constant v20::v17::<v10, v2>::{constant#0}
        //~| ERROR maximum number of nodes exceeded in constant v20::v17::<v10, v2>::{constant#0}
        //~| ERROR maximum number of nodes exceeded in constant v20::v17::<v10, v2>::{constant#0}
        pub const fn v21() -> v18 {
            //~^ ERROR cannot find type `v18` in this scope
            v18 { _p: () }
            //~^ ERROR cannot find struct, variant or union type `v18` in this scope
        }
    }
}
pub use v20::{v13, v17};
//~^ ERROR unresolved import `v20::v13`
fn main() {}
