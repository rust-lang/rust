// rustc-env:RUST_MIN_STACK=1048576
// normalize-stderr-test: "long-type-\d+" -> "long-type-hash"

#![recursion_limit = "2048"]

trait MyTrait {}

impl<'a> MyTrait for &'a () {}
impl<T> MyTrait for &'_ (T,) where for<'a> &'a T: MyTrait {}

fn of<'a, T: 'a>() -> T
where
    &'a T: MyTrait,
{
    todo!()
}

fn main() {
    let _x: () = of(); //~ ERROR overflow evaluating the requirement `for<'a> &'a (_,): MyTrait`
}
