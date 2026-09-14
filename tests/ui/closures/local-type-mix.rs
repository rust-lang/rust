// Check that using the parameter name in its type does not ICE.
//@ edition:2018

fn main() {
    let _ = |x: x| x; //~ ERROR cannot find type `x` in this scope
    let _ = |x: bool| -> x { x }; //~ ERROR cannot find type `x` in this scope
    let _ = async move |x: x| x; //~ ERROR cannot find type `x` in this scope
    let _ = async move |x: bool| -> x { x }; //~ ERROR cannot find type `x` in this scope
}

fn foo(x: x) {} //~ ERROR cannot find type `x` in this scope
fn foo_ret(x: bool) -> x {} //~ ERROR cannot find type `x` in this scope

async fn async_foo(x: x) {} //~ ERROR cannot find type `x` in this scope
async fn async_foo_ret(x: bool) -> x {} //~ ERROR cannot find type `x` in this scope
