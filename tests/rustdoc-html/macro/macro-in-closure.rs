// Regression issue for rustdoc ICE encountered in PR #65252.

#![feature(decl_macro)]

fn main() {
    || {
        macro m() {}
    };

    let _ = || {
        macro n() {}
    };

    let cond = true;
    let _ = || if cond { macro n() {} } else { panic!() };
}
