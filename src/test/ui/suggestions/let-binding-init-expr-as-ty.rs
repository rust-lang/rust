pub fn foo(num: i32) -> i32 { //~ ERROR mismatched types
    let foo: i32::from_be(num);
    //~^ ERROR expected type, found local variable `num`
    //~| ERROR parenthesized type parameters may only be used with a `Fn` trait
    //~| ERROR ambiguous associated type
    //~| WARNING this was previously accepted by the compiler but is being phased out
}

fn main() {
    let _ = foo(42);
}
