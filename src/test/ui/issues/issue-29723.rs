#![feature(nll)]

// test for https://github.com/rust-lang/rust/issues/29723

fn main() {
    let s = String::new();
    let _s = match 0 {
        0 if { drop(s); false } => String::from("oops"),
        _ => {
            // This should trigger an error,
            // s could have been moved from.
            s
            //~^ ERROR use of moved value: `s`
        }
    };
}
