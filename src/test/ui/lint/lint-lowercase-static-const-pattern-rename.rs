// build-pass (FIXME(62277): could be check-pass?)
// Issue #7526: lowercase static constants in patterns look like bindings

// This is similar to lint-lowercase-static-const-pattern.rs, except it
// shows the expected usual workaround (choosing a different name for
// the static definition) and also demonstrates that one can work
// around this problem locally by renaming the constant in the `use`
// form to an uppercase identifier that placates the lint.

#![deny(non_upper_case_globals)]

pub const A : isize = 97;

fn f() {
    let r = match (0,0) {
        (0, A) => 0,
        (x, y) => 1 + x + y,
    };
    assert_eq!(r, 1);
    let r = match (0,97) {
        (0, A) => 0,
        (x, y) => 1 + x + y,
    };
    assert_eq!(r, 0);
}

mod m {
    #[allow(non_upper_case_globals)]
    pub const aha : isize = 7;
}

fn g() {
    use self::m::aha as AHA;
    let r = match (0,0) {
        (0, AHA) => 0,
        (x, y)   => 1 + x + y,
    };
    assert_eq!(r, 1);
    let r = match (0,7) {
        (0, AHA) => 0,
        (x, y)   => 1 + x + y,
    };
    assert_eq!(r, 0);
}

fn h() {
    let r = match (0,0) {
        (0, self::m::aha) => 0,
        (x, y)      => 1 + x + y,
    };
    assert_eq!(r, 1);
    let r = match (0,7) {
        (0, self::m::aha) => 0,
        (x, y)      => 1 + x + y,
    };
    assert_eq!(r, 0);
}

pub fn main () {
    f();
    g();
    h();
}
