//@ edition:2021

struct StructA {
    b: StructB,
}

async fn spawn_blocking<T>(f: impl (Fn() -> T) + Send + Sync + 'static) -> T {
    todo!()
}

impl StructA {
    async fn foo(&self) {
        let bar = self.b.bar().await;
        spawn_blocking(move || {
            //~^ ERROR borrowed data escapes outside of method
            self.b;
            //~^ ERROR cannot move out of `self.b`, as `self` is a captured variable in an `Fn` closure
        })
        .await;
    }
}

struct StructB {}

impl StructB {
    async fn bar(&self) -> Option<u8> {
        None
    }
}

fn main() {}
