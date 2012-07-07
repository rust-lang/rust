// error-pattern:import

import y::x;

mod y {
    import x;
    export x;
}

fn main() { }
