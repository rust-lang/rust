// edition:2018
trait T {
    async fn foo() {} //~ ERROR trait fns cannot be declared `async`
    async fn bar(&self) {} //~ ERROR trait fns cannot be declared `async`
}

fn main() {}
