// compile-flags: -Z parse-only

fn f() {
    let x: fn~() = || (); //~ ERROR expected `(`, found `~`
}
