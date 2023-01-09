// test for https://github.com/rust-lang/rust/issues/29723

#![feature(if_let_guard)]

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

    let s = String::new();
    let _s = match 0 {
        0 if let Some(()) = { drop(s); None } => String::from("oops"),
        _ => s //~ ERROR use of moved value: `s`
    };
}
