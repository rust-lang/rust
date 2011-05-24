// error-pattern: cyclic import

import y::x;

mod y {
    import x;
}

fn main() {
}
