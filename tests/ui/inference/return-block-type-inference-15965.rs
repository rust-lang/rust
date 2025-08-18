//! Regression test for https://github.com/rust-lang/rust/issues/15965

fn main() {
    return
        { return () }
//~^ ERROR type annotations needed [E0282]
    ()
    ;
}
