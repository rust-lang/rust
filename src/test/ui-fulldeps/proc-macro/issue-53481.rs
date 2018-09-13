// compile-pass
// aux-build:issue-53481.rs

#[macro_use]
extern crate issue_53481;

mod m1 {
    use m2::MyTrait;

    #[derive(MyTrait)]
    struct A {}
}

mod m2 {
    pub type MyTrait = u8;

    #[derive(MyTrait)]
    #[my_attr]
    struct B {}
}

fn main() {}
