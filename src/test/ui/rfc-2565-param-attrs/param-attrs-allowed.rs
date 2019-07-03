// compile-flags: --cfg something
// build-pass (FIXME(62277): could be check-pass?)

#![feature(param_attrs)]

extern "C" {
    fn ffi(
        #[allow(C)] a: i32,
        #[cfg(something)] b: i32,
        #[cfg_attr(something, cfg(nothing))] c: i32,
        #[deny(C)] d: i32,
        #[forbid(C)] #[warn(C)] ...
    );
}

type FnType = fn(
    #[allow(C)] a: i32,
    #[cfg(something)] b: i32,
    #[cfg_attr(something, cfg(nothing))] c: i32,
    #[deny(C)] d: i32,
    #[forbid(C)] #[warn(C)] e: i32
);

pub fn foo(
    #[allow(C)] a: i32,
    #[cfg(something)] b: i32,
    #[cfg_attr(something, cfg(nothing))] c: i32,
    #[deny(C)] d: i32,
    #[forbid(C)] #[warn(C)] e: i32
) {}

// self, &self and &mut self

struct SelfStruct {}
impl SelfStruct {
    fn foo(
        #[allow(C)] self,
        #[cfg(something)] a: i32,
        #[cfg_attr(something, cfg(nothing))]
        #[deny(C)] b: i32,
    ) {}
}

struct RefStruct {}
impl RefStruct {
    fn foo(
        #[allow(C)] &self,
        #[cfg(something)] a: i32,
        #[cfg_attr(something, cfg(nothing))]
        #[deny(C)] b: i32,
    ) {}
}
trait RefTrait {
    fn foo(
        #[forbid(C)] &self,
        #[warn(C)] a: i32
    ) {}
}
impl RefTrait for RefStruct {
    fn foo(
        #[forbid(C)] &self,
        #[warn(C)] a: i32
    ) {}
}

struct MutStruct {}
impl MutStruct {
    fn foo(
        #[allow(C)] &mut self,
        #[cfg(something)] a: i32,
        #[cfg_attr(something, cfg(nothing))]
        #[deny(C)] b: i32,
    ) {}
}
trait MutTrait {
    fn foo(
        #[forbid(C)] &mut self,
        #[warn(C)] a: i32
    ) {}
}
impl MutTrait for MutStruct {
    fn foo(
        #[forbid(C)] &mut self,
        #[warn(C)] a: i32
    ) {}
}

// self: Self, self: &Self and self: &mut Self

struct NamedSelfSelfStruct {}
impl NamedSelfSelfStruct {
    fn foo(
        #[allow(C)] self: Self,
        #[cfg(something)] a: i32,
        #[cfg_attr(something, cfg(nothing))]
        #[deny(C)] b: i32,
    ) {}
}

struct NamedSelfRefStruct {}
impl NamedSelfRefStruct {
    fn foo(
        #[allow(C)] self: &Self,
        #[cfg(something)] a: i32,
        #[cfg_attr(something, cfg(nothing))]
        #[deny(C)] b: i32,
    ) {}
}
trait NamedSelfRefTrait {
    fn foo(
        #[forbid(C)] self: &Self,
        #[warn(C)] a: i32
    ) {}
}
impl NamedSelfRefTrait for NamedSelfRefStruct {
    fn foo(
        #[forbid(C)] self: &Self,
        #[warn(C)] a: i32
    ) {}
}

struct NamedSelfMutStruct {}
impl NamedSelfMutStruct {
    fn foo(
        #[allow(C)] self: &mut Self,
        #[cfg(something)] a: i32,
        #[cfg_attr(something, cfg(nothing))]
        #[deny(C)] b: i32,
    ) {}
}
trait NamedSelfMutTrait {
    fn foo(
        #[forbid(C)] self: &mut Self,
        #[warn(C)] a: i32
    ) {}
}
impl NamedSelfMutTrait for NamedSelfMutStruct {
    fn foo(
        #[forbid(C)] self: &mut Self,
        #[warn(C)] a: i32
    ) {}
}

// &'a self and &'a mut self

struct NamedLifetimeRefStruct {}
impl NamedLifetimeRefStruct {
    fn foo<'a>(
        #[allow(C)] self: &'a Self,
        #[cfg(something)] a: i32,
        #[cfg_attr(something, cfg(nothing))]
        #[deny(C)] b: i32,
    ) {}
}
trait NamedLifetimeRefTrait {
    fn foo<'a>(
        #[forbid(C)] &'a self,
        #[warn(C)] a: i32
    ) {}
}
impl NamedLifetimeRefTrait for NamedLifetimeRefStruct {
    fn foo<'a>(
        #[forbid(C)] &'a self,
        #[warn(C)] a: i32
    ) {}
}

struct NamedLifetimeMutStruct {}
impl NamedLifetimeMutStruct {
    fn foo<'a>(
        #[allow(C)] self: &'a mut Self,
        #[cfg(something)] a: i32,
        #[cfg_attr(something, cfg(nothing))]
        #[deny(C)] b: i32,
    ) {}
}
trait NamedLifetimeMutTrait {
    fn foo<'a>(
        #[forbid(C)] &'a mut self,
        #[warn(C)] a: i32
    ) {}
}
impl NamedLifetimeMutTrait for NamedLifetimeMutStruct {
    fn foo<'a>(
        #[forbid(C)] &'a mut self,
        #[warn(C)] a: i32
    ) {}
}

// Box<Self>

struct BoxSelfStruct {}
impl BoxSelfStruct {
    fn foo(
        #[allow(C)] self: Box<Self>,
        #[cfg(something)] a: i32,
        #[cfg_attr(something, cfg(nothing))]
        #[deny(C)] b: i32,
    ) {}
}
trait BoxSelfTrait {
    fn foo(
        #[forbid(C)] self: Box<Self>,
        #[warn(C)] a: i32
    ) {}
}
impl BoxSelfTrait for BoxSelfStruct {
    fn foo(
        #[forbid(C)] self: Box<Self>,
        #[warn(C)] a: i32
    ) {}
}

fn main() {
    let _: unsafe extern "C" fn(_, _, _, ...) = ffi;
    let _: fn(_, _, _, _) = foo;
    let _: FnType = |_, _, _, _| {};
    let c = |
        #[allow(C)] a: u32,
        #[cfg(something)] b: i32,
        #[cfg_attr(something, cfg(nothing))]
        #[deny(C)] c: i32,
    | {};
    let _ = c(1, 2);
}
