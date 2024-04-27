pub fn foo(num: i32) -> i32 {
    let foo: i32::from_be(num);
    //~^ ERROR expected type, found local variable `num`
    //~| ERROR parenthesized type parameters may only be used with a `Fn` trait
    //~| ERROR ambiguous associated type
    foo
}

fn main() {
    let _ = foo(42);
}
