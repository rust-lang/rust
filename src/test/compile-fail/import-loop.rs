// error-pattern: cyclic import

import y::x;

mod y {
    import x;
    export x;
}

fn main() { }
