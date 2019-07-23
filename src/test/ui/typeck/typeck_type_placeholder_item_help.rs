// This test checks that it proper item type will be suggested when
// using the `_` type placeholder.

fn test1() -> _ { Some(42) }
//~^ ERROR the type placeholder `_` is not allowed within types on item signatures

const TEST2: _ = 42u32;
//~^ ERROR the type placeholder `_` is not allowed within types on item signatures

const TEST3: _ = Some(42);
//~^ ERROR the type placeholder `_` is not allowed within types on item signatures

trait Test4 {
    const TEST4: _ = 42;
    //~^ ERROR the type placeholder `_` is not allowed within types on item signatures
}

struct Test5;

impl Test5 {
    const TEST5: _ = 13;
    //~^ ERROR the type placeholder `_` is not allowed within types on item signatures
}

pub fn main() {
    let _: Option<usize> = test1();
    let _: f64 = test1();
    let _: Option<i32> = test1();
}
