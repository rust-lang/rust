// error-pattern:import

use y::x;

mod y {
    pub use y::x;
}

fn main() { }
