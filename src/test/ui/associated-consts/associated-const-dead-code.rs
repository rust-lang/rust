#![deny(dead_code)]

struct MyFoo;

impl MyFoo {
    const BAR: u32 = 1;
    //~^ ERROR associated constant is never used: `BAR`
}

fn main() {
    let _: MyFoo = MyFoo;
}
