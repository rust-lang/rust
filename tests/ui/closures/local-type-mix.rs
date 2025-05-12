// Check that using the parameter name in its type does not ICE.
//@ edition:2018

fn main() {
    let _ = |x: x| x; //~ ERROR expected type
    let _ = |x: bool| -> x { x }; //~ ERROR expected type
    let _ = async move |x: x| x; //~ ERROR expected type
    let _ = async move |x: bool| -> x { x }; //~ ERROR expected type
}

fn foo(x: x) {} //~ ERROR expected type
fn foo_ret(x: bool) -> x {} //~ ERROR expected type

async fn async_foo(x: x) {} //~ ERROR expected type
async fn async_foo_ret(x: bool) -> x {} //~ ERROR expected type
