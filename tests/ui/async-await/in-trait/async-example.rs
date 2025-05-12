//@ check-pass
//@ edition: 2021

trait MyTrait {
    #[allow(async_fn_in_trait)]
    async fn foo(&self) -> i32;

    #[allow(async_fn_in_trait)]
    async fn bar(&self) -> i32;
}

impl MyTrait for i32 {
    async fn foo(&self) -> i32 {
        *self
    }

    async fn bar(&self) -> i32 {
        self.foo().await
    }
}

fn main() {
    let x = 5;
    // Calling from non-async context
    let _ = x.foo();
    let _ = x.bar();
    // Calling from async block in non-async context
    async {
        let _: i32 = x.foo().await;
        let _: i32 = x.bar().await;
    };
}
