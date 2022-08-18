// edition:2021

trait T {
    async fn foo(); //~ ERROR functions in traits cannot be declared `async`
}

fn main() {}
