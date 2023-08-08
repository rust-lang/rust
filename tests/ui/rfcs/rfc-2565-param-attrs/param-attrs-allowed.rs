// check-pass
// compile-flags: --cfg something

#![deny(unused_mut)]

extern "C" {
    fn ffi(
        #[allow(unused_mut)] a: i32,
        #[cfg(something)] b: i32,
        #[cfg_attr(something, cfg(nothing))] c: i32,
        #[forbid(unused_mut)] d: i32,
        #[deny(unused_mut)] #[warn(unused_mut)] ...
    );
}

type FnType = fn(
    #[allow(unused_mut)] a: i32,
    #[cfg(something)] b: i32,
    #[cfg_attr(something, cfg(nothing))] c: i32,
    #[forbid(unused_mut)] d: i32,
    #[deny(unused_mut)] #[warn(unused_mut)] e: i32
);

pub fn foo(
    #[allow(unused_mut)] a: i32,
    #[cfg(something)] b: i32,
    #[cfg_attr(something, cfg(nothing))] c: i32,
    #[forbid(unused_mut)] d: i32,
    #[deny(unused_mut)] #[warn(unused_mut)] _e: i32
) {}

// self

struct SelfStruct {}
impl SelfStruct {
    fn foo(
        #[allow(unused_mut)] self,
        #[cfg(something)] a: i32,
        #[cfg_attr(something, cfg(nothing))]
        #[deny(unused_mut)] b: i32,
    ) {}
}

struct RefStruct {}
impl RefStruct {
    fn foo(
        #[allow(unused_mut)] &self,
        #[cfg(something)] a: i32,
        #[cfg_attr(something, cfg(nothing))]
        #[deny(unused_mut)] b: i32,
    ) {}
}
trait RefTrait {
    fn foo(
        #[forbid(unused_mut)] &self,
        #[warn(unused_mut)] a: i32
    ) {}
}
impl RefTrait for RefStruct {
    fn foo(
        #[forbid(unused_mut)] &self,
        #[warn(unused_mut)] a: i32
    ) {}
}

// Box<Self>

struct BoxSelfStruct {}
impl BoxSelfStruct {
    fn foo(
        #[allow(unused_mut)] self: Box<Self>,
        #[cfg(something)] a: i32,
        #[cfg_attr(something, cfg(nothing))]
        #[deny(unused_mut)] b: i32,
    ) {}
}
trait BoxSelfTrait {
    fn foo(
        #[forbid(unused_mut)] self: Box<Self>,
        #[warn(unused_mut)] a: i32
    ) {}
}
impl BoxSelfTrait for BoxSelfStruct {
    fn foo(
        #[forbid(unused_mut)] self: Box<Self>,
        #[warn(unused_mut)] a: i32
    ) {}
}

fn main() {
    let _: unsafe extern "C" fn(_, _, _, ...) = ffi;
    let _: fn(_, _, _, _) = foo;
    let _: FnType = |_, _, _, _| {};
    let c = |
        #[allow(unused_mut)] a: u32,
        #[cfg(something)] b: i32,
        #[cfg_attr(something, cfg(nothing))]
        #[deny(unused_mut)] c: i32,
    | {};
    c(1, 2);
}
