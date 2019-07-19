// This test checks that it proper item type will be suggested when
// using the `_` type placeholder.

fn test1() -> _ { Some(42) }
//~^ ERROR the type placeholder `_` is not allowed within types on item signatures

pub fn main() {
    let _: Option<usize> = test1();
    let _: f64 = test1();
    let _: Option<i32> = test1();
}
