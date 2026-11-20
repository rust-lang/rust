#![deny(unused)]
//~^ NOTE lint level is defined here

#[must_use = "AAA"]
//~^ NOTE also specified here
#[must_use = "BBB"]
//~^ ERROR unused attribute
//~| WARN previously accepted
//~| NOTE `#[deny(unused_attributes)]` implied by `#[deny(unused)]`
fn must_use() -> usize {
    0
}

fn main() {
    must_use();
    //~^ ERROR unused return value of `must_use` that must be used
    //~| NOTE AAA
    //~| NOTE `#[deny(unused_must_use)]` implied by `#[deny(unused)]`
}
