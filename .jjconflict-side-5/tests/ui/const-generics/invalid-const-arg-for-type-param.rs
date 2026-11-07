use std::convert::TryInto;

struct S;

fn main() {
    let _: u32 = 5i32.try_into::<32>().unwrap();
    //~^ ERROR method takes

    S.f::<0>();
    //~^ ERROR no method named `f`

    S::<0>;
    //~^ ERROR struct takes 0
}
