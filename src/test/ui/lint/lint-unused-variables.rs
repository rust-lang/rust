// compile-flags: --cfg something
// edition:2018

#![feature(async_await, async_closure, param_attrs)]
#![deny(unused_variables)]

async fn foo_async(
    a: i32,
    //~^ ERROR unused variable: `a`
    #[allow(unused_variables)] b: i32,
) {}
fn foo(
    #[allow(unused_variables)] a: i32,
    b: i32,
    //~^ ERROR unused variable: `b`
) {}

struct RefStruct {}
impl RefStruct {
    async fn bar_async(
        &self,
        a: i32,
        //~^ ERROR unused variable: `a`
        #[allow(unused_variables)] b: i32,
    ) {}
    fn bar(
        &self,
        #[allow(unused_variables)] a: i32,
        b: i32,
        //~^ ERROR unused variable: `b`
    ) {}
}
trait RefTrait {
    fn bar(
        &self,
        #[allow(unused_variables)] a: i32,
        b: i32,
        //~^ ERROR unused variable: `b`
    ) {}
}
impl RefTrait for RefStruct {
    fn bar(
        &self,
        #[allow(unused_variables)] a: i32,
        b: i32,
        //~^ ERROR unused variable: `b`
    ) {}
}

fn main() {
    let _: fn(_, _) = foo;
    let a = async move |
        a: i32,
        //~^ ERROR unused variable: `a`
        #[allow(unused_variables)] b: i32,
    | {};
    let b = |
        #[allow(unused_variables)] a: i32,
        b: i32,
        //~^ ERROR unused variable: `b`
    | {};
    let _ = a(1, 2);
    let _ = b(1, 2);
}
