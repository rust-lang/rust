// edition:2015

async fn foo() {} //~ ERROR `async fn` is not permitted in the 2015 edition
                  //~^ ERROR async fn is unstable

fn main() {
    let _ = async {}; //~ ERROR cannot find struct, variant or union type `async`
    let _ = async || { true }; //~ ERROR cannot find value `async` in this scope
}
