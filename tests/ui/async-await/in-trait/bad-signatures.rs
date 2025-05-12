//@ edition:2021


trait MyTrait {
    async fn bar(&abc self);
    //~^ ERROR expected identifier, found keyword `self`
    //~| ERROR expected one of `:`, `@`, or `|`, found keyword `self`
}

impl MyTrait for () {
    async fn bar(&self) {}
}

fn main() {}
