// This test ensures that using `auto_cfg(show(key))` works correctly.

#![feature(doc_cfg)]
#![crate_name = "foo"]

#![doc(auto_cfg(hide(meow)))]
#![doc(auto_cfg(hide(meow, values("lol"))))]

//@ has foo/fn.foo.html
//@ has - '//*[@class="item-info"]/*[@class="stab portability"]' 'non-blob'
//@ !has - '//*[@class="item-info"]/*[@class="stab portability"]' 'non-meow'
#[cfg(not(meow))]
#[cfg(not(blob))]
pub fn foo() {}

//@ has foo/fn.bar.html
//@ has - '//*[@class="item-info"]/*[@class="stab portability"]' 'non-blob=lol'
//@ !has - '//*[@class="item-info"]/*[@class="stab portability"]' 'non-meow=lol'
#[cfg(not(meow = "lol"))]
#[cfg(not(blob = "lol"))]
pub fn bar() {}

//@ has foo/fn.babar.html
//@ has - '//*[@class="item-info"]/*[@class="stab portability"]' 'non-blob=lola'
//@ has - '//*[@class="item-info"]/*[@class="stab portability"]' 'non-meow=lola'
#[cfg(not(meow = "lola"))]
#[cfg(not(blob = "lola"))]
pub fn babar() {}

pub mod sub {
    // We show again `meow`, however `meow="lol"` should still be hidden.
    #![doc(auto_cfg(show(meow)))]

    //@ has foo/sub/fn.foo.html
    //@ has - '//*[@class="item-info"]/*[@class="stab portability"]' 'non-blob'
    //@ has - '//*[@class="item-info"]/*[@class="stab portability"]' 'non-meow'
    #[cfg(not(meow))]
    #[cfg(not(blob))]
    pub fn foo() {}

    //@ has foo/sub/fn.bar.html
    //@ has - '//*[@class="item-info"]/*[@class="stab portability"]' 'non-blob=lol'
    //@ !has - '//*[@class="item-info"]/*[@class="stab portability"]' 'non-meow=lol'
    #[cfg(not(meow = "lol"))]
    #[cfg(not(blob = "lol"))]
    pub fn bar() {}

    //@ has foo/sub/fn.babar.html
    //@ has - '//*[@class="item-info"]/*[@class="stab portability"]' 'non-blob=lola'
    //@ has - '//*[@class="item-info"]/*[@class="stab portability"]' 'non-meow=lola'
    #[cfg(not(meow = "lola"))]
    #[cfg(not(blob = "lola"))]
    pub fn babar() {}
}

pub mod sub2 {
    // We show again `meow = "lol`, however `meow` should still be hidden.
    #![doc(auto_cfg(show(meow, values("lol"))))]

    //@ has foo/sub2/fn.foo.html
    //@ has - '//*[@class="item-info"]/*[@class="stab portability"]' 'non-blob'
    //@ !has - '//*[@class="item-info"]/*[@class="stab portability"]' 'non-meow'
    #[cfg(not(meow))]
    #[cfg(not(blob))]
    pub fn foo() {}

    //@ has foo/sub2/fn.bar.html
    //@ has - '//*[@class="item-info"]/*[@class="stab portability"]' 'non-blob=lol'
    //@ has - '//*[@class="item-info"]/*[@class="stab portability"]' 'non-meow=lol'
    #[cfg(not(meow = "lol"))]
    #[cfg(not(blob = "lol"))]
    pub fn bar() {}

    //@ has foo/sub2/fn.babar.html
    //@ has - '//*[@class="item-info"]/*[@class="stab portability"]' 'non-blob=lola'
    //@ has - '//*[@class="item-info"]/*[@class="stab portability"]' 'non-meow=lola'
    #[cfg(not(meow = "lola"))]
    #[cfg(not(blob = "lola"))]
    pub fn babar() {}
}

pub mod sub3 {
    // We show again `meow = "lol`, but by using `any()` this time.
    #![doc(auto_cfg(show(meow, values(any()))))]

    //@ has foo/sub3/fn.foo.html
    //@ has - '//*[@class="item-info"]/*[@class="stab portability"]' 'non-blob'
    //@ !has - '//*[@class="item-info"]/*[@class="stab portability"]' 'non-meow'
    #[cfg(not(meow))]
    #[cfg(not(blob))]
    pub fn foo() {}

    //@ has foo/sub3/fn.bar.html
    //@ has - '//*[@class="item-info"]/*[@class="stab portability"]' 'non-blob=lol'
    //@ has - '//*[@class="item-info"]/*[@class="stab portability"]' 'non-meow=lol'
    #[cfg(not(meow = "lol"))]
    #[cfg(not(blob = "lol"))]
    pub fn bar() {}

    //@ has foo/sub3/fn.babar.html
    //@ has - '//*[@class="item-info"]/*[@class="stab portability"]' 'non-blob=lola'
    //@ has - '//*[@class="item-info"]/*[@class="stab portability"]' 'non-meow=lola'
    #[cfg(not(meow = "lola"))]
    #[cfg(not(blob = "lola"))]
    pub fn babar() {}
}
