//! Arrays created with `[value; length]` syntax need the length to be known at
//! compile time. This test makes sure the compiler rejects runtime values like
//! function parameters in the length position.

fn main() {
    fn create_array(n: usize) {
        let _x = [0; n];
        //~^ ERROR attempt to use a non-constant value in a constant [E0435]
    }
}
