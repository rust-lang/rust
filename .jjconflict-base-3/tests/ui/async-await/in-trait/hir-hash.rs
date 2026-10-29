// Issue #122508

//@ check-pass
//@ incremental
//@ edition:2021

trait MyTrait {
    async fn bar(&self) -> i32;
}

fn main() {}
