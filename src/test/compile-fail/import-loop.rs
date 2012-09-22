// error-pattern:import

use y::x;

mod y {
    #[legacy_exports];
    import x;
    export x;
}

fn main() { }
