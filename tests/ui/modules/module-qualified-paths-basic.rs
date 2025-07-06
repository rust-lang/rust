//! Checks that functions from different modules are accessible via their fully-qualified paths.

//@ run-pass

mod foo {
    pub fn x() -> isize {
        return 1;
    }
}

mod bar {
    pub fn y() -> isize {
        return 1;
    }
}

pub fn main() {
    foo::x();
    bar::y();
}
