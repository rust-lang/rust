// compile-flags: --cfg something

#![feature(param_attrs)]
#![deny(unused_variables)]

extern "C" {
    fn ffi(
        #[cfg(nothing)] a: i32,
        #[cfg(something)] b: i32,
        #[cfg_attr(something, cfg(nothing))] c: i32,
        #[cfg_attr(nothing, cfg(nothing))] ...
    );
}

type FnType = fn(
    #[cfg(nothing)] a: i32,
    #[cfg(something)] b: i32,
    #[cfg_attr(nothing, cfg(nothing))] c: i32,
    #[cfg_attr(something, cfg(nothing))] d: i32,
);

fn foo(
    #[cfg(nothing)] a: i32,
    #[cfg(something)] b: i32,
    //~^ ERROR unused variable: `b` [unused_variables]
    #[cfg_attr(nothing, cfg(nothing))] c: i32,
    //~^ ERROR unused variable: `c` [unused_variables]
    #[cfg_attr(something, cfg(nothing))] d: i32,
) {}

struct RefStruct {}
impl RefStruct {
    fn bar(
        &self,
        #[cfg(nothing)] a: i32,
        #[cfg(something)] b: i32,
        //~^ ERROR unused variable: `b` [unused_variables]
        #[cfg_attr(nothing, cfg(nothing))] c: i32,
        //~^ ERROR unused variable: `c` [unused_variables]
        #[cfg_attr(something, cfg(nothing))] d: i32,
    ) {}
}
trait RefTrait {
    fn bar(
        &self,
        #[cfg(nothing)] a: i32,
        #[cfg(something)] b: i32,
        //~^ ERROR unused variable: `b` [unused_variables]
        #[cfg_attr(nothing, cfg(nothing))] c: i32,
        //~^ ERROR unused variable: `c` [unused_variables]
        #[cfg_attr(something, cfg(nothing))] d: i32,
    ) {}
}
impl RefTrait for RefStruct {
    fn bar(
        &self,
        #[cfg(nothing)] a: i32,
        #[cfg(something)] b: i32,
        //~^ ERROR unused variable: `b` [unused_variables]
        #[cfg_attr(nothing, cfg(nothing))] c: i32,
        //~^ ERROR unused variable: `c` [unused_variables]
        #[cfg_attr(something, cfg(nothing))] d: i32,
    ) {}
}

fn main() {
    let _: unsafe extern "C" fn(_, ...) = ffi;
    let _: fn(_, _) = foo;
    let _: FnType = |_, _| {};
    let c = |
        #[cfg(nothing)] a: i32,
        #[cfg(something)] b: i32,
        //~^ ERROR unused variable: `b` [unused_variables]
        #[cfg_attr(nothing, cfg(nothing))] c: i32,
        //~^ ERROR unused variable: `c` [unused_variables]
        #[cfg_attr(something, cfg(nothing))] d: i32,
    | {};
    let _ = c(1, 2);
}
