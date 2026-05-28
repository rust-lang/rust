//@ edition:2021


trait MyTrait {
    async fn foo<'a>(&self);
    async fn bar(&self);
}

impl MyTrait for i32 {
    async fn foo(&self) {}
    //~^ ERROR lifetime parameters or bounds on method `foo` do not match the trait declaration

    async fn bar(&self) {
        self.foo();
    }
}

fn main() {}
