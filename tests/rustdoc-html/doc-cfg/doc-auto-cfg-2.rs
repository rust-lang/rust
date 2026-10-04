// Basic tests covering RFC 3631 features.

#![crate_name = "foo"]
#![feature(doc_cfg)]
#![doc(auto_cfg(hide(feature, values("hidden"))))]

//@ has 'foo/index.html'
//@ !has - '//*[@class="stab portability"]' 'Non-moustache'
//@ has - '//*[@class="stab portability"]' 'Non-pistache'
//@ count - '//*[@class="stab portability"]' 1

//@ has 'foo/m/index.html'
//@ count - '//*[@title="Available on non-crate feature `hidden` only"]' 2
#[cfg(not(feature = "hidden"))]
pub mod m {
    //@ count 'foo/m/type.A.html' '//*[@class="stab portability"]' 0
    pub struct A;

    //@ has 'foo/m/inner/index.html' '//*[@class="stab portability"]' 'Available on non-crate feature hidden only.'
    #[doc(auto_cfg(show(feature, values("hidden"))))]
    pub mod inner {
        //@ has 'foo/m/inner/type.B.html' '//*[@class="stab portability"]' 'Available on non-crate feature hidden only.'
        pub struct B;

        //@ count 'foo/m/inner/type.A.html' '//*[@class="stab portability"]' 0
        #[doc(auto_cfg(hide(feature, values("hidden"))))]
        pub struct A;
    }

    //@ has 'foo/m/type.B.html' '//*[@class="stab portability"]' 'Available on non-crate feature hidden only.'
    #[doc(auto_cfg(show(feature, values("hidden"))))]
    pub struct B;
}

//@ count 'foo/n/index.html' '//*[@title="Available on non-crate feature `moustache` only"]' 3
//@ count - '//dl/dt' 4
#[cfg(not(feature = "moustache"))]
#[doc(auto_cfg = false)]
pub mod n {
    // Should not have `moustache` listed.
    //@ count 'foo/n/type.X.html' '//*[@class="stab portability"]' 0
    pub struct X;

    // Should re-enable `auto_cfg` and make `moustache` listed.
    //@ has 'foo/n/type.Y.html' '//*[@class="stab portability"]' \
    //  'Available on non-crate feature moustache only.'
    #[doc(auto_cfg)]
    pub struct Y;

    // Should re-enable `auto_cfg` and make `moustache` listed for itself
    // and for `Y`.
    //@ has 'foo/n/inner/index.html' '//*[@class="stab portability"]' \
    //  'Available on non-crate feature moustache only.'
    #[doc(auto_cfg = true)]
    pub mod inner {
        //@ has 'foo/n/inner/type.Y.html' '//*[@class="stab portability"]' \
        //  'Available on non-crate feature moustache only.'
        pub struct Y;
    }

    // Should re-enable `auto_cfg` and make `moustache` listed.
    //@ has 'foo/n/type.Z.html' '//*[@class="stab portability"]' \
    //  'Available on non-crate feature moustache only.'
    #[doc(auto_cfg(hide(feature, values("hidden"))))]
    pub struct Z;
}

// Checking inheritance.
//@ has 'foo/o/index.html' '//*[@class="stab portability"]' \
//  'Available on non-crate feature pistache only.'
#[doc(cfg(not(feature = "pistache")))]
pub mod o {
    //@ has 'foo/o/type.A.html' '//*[@class="stab portability"]' \
    //  'Available on non-crate feature pistache and non-crate feature tarte only.'
    #[doc(cfg(not(feature = "tarte")))]
    pub struct A;
}
