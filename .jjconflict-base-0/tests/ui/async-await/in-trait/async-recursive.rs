//@ edition: 2021

trait MyTrait {
    async fn foo_recursive(&self, n: usize) -> i32;
}

impl MyTrait for i32 {
    async fn foo_recursive(&self, n: usize) -> i32 {
        //~^ ERROR recursion in an async fn requires boxing
        if n > 0 {
            self.foo_recursive(n - 1).await
        } else {
            *self
        }
    }
}

fn main() {}
