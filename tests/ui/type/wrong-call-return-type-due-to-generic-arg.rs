fn function<T>(x: T, y: bool) -> T {
    x
}

struct S {}
impl S {
    fn method<T>(&self, x: T) -> T {
        x
    }
}

fn wrong_arg_type(x: u32) -> u32 {
    x
}

fn main() {
    // Should not trigger.
    let x = wrong_arg_type(0u16); //~ ERROR mismatched types
    let x: u16 = function(0, 0u8); //~ ERROR mismatched types

    // Should trigger exactly once for the first argument.
    let x: u16 = function(0u32, 0u8); //~ ERROR arguments to this function are incorrect

    // Should trigger.
    let x: u16 = function(0u32, true); //~ ERROR mismatched types
    let x: u16 = (S {}).method(0u32); //~ ERROR mismatched types
    function(0u32, 8u8) //~ ERROR arguments to this function are incorrect
}
