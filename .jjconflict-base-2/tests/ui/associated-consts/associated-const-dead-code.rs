#![deny(dead_code)]

struct MyFoo;

impl MyFoo {
    const BAR: u32 = 1;
    //~^ ERROR associated constant `BAR` is never used
}

fn main() {
    let _: MyFoo = MyFoo;
}
