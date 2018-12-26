#![feature(main)]

#[main]
fn main1() {
}

mod foo {
    #[main]
    fn main2() { //~ ERROR multiple functions with a #[main] attribute
    }
}
