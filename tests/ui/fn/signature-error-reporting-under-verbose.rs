// compile-flags: -Zverbose

fn foo(_: i32, _: i32) {}

fn needs_ptr(_: fn(i32, u32)) {}
//~^ NOTE function defined here
//~| NOTE

fn main() {
    needs_ptr(foo);
    //~^ ERROR mismatched types
    //~| NOTE expected fn pointer, found fn item
    //~| NOTE expected fn pointer `fn(i32, u32)`
    //~| NOTE arguments to this function are incorrect
    //~| NOTE when the arguments and return types match, functions can be coerced to function pointers
}
