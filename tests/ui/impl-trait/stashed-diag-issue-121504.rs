//@ edition: 2021

trait MyTrait {
    async fn foo(self) -> (Self, i32);
}

impl MyTrait for xyz::T { //~ ERROR cannot find item `xyz`
    async fn foo(self, key: i32) -> (u32, i32) {
        (self, key)
    }
}

fn main() {}
