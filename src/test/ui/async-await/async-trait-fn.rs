// edition:2018
trait T {
    async fn foo() {} //~ ERROR functions in traits cannot be declared `async`
    async fn bar(&self) {} //~ ERROR functions in traits cannot be declared `async`
}

fn main() {}
