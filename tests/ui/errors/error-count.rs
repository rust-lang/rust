//! Regression test for <https://github.com/rust-lang/rust/issues/33525>.
//! Test rustc emits right error count.
//! (this used to return `aborting due to 2 previous errors`)

fn main() {
    a; //~ ERROR cannot find value `a`
    "".lorem; //~ ERROR no field
    "".ipsum; //~ ERROR no field
}
