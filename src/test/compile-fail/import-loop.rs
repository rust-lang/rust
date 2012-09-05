// error-pattern:import

use y::x;

mod y {
    import x;
    export x;
}

fn main() { }
