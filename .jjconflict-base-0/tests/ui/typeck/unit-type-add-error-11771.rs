//! Regression test for https://github.com/rust-lang/rust/issues/11771

fn main() {
    let x = ();
    1 +
    x //~^ ERROR E0277
    ;

    let x: () = ();
    1 +
    x //~^ ERROR E0277
    ;
}
