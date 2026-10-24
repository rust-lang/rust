//! Don't suggest breaking with value from `for` or `while` loops
//!
//! Regression test for https://github.com/rust-lang/rust/issues/150850

fn returns_i32() -> i32 { 0 }

fn suggest_breaking_from_loop() {
    let _ = loop {
        returns_i32() //~ ERROR mismatched types
        //~^ SUGGESTION ;
        //~| SUGGESTION break
    };
}

fn dont_suggest_breaking_from_for() {
    let _ = for _ in 0.. {
        returns_i32() //~ ERROR mismatched types
        //~^ SUGGESTION ;
    };
}

fn dont_suggest_breaking_from_while() {
    let cond = true;
    let _ = while cond {
        returns_i32() //~ ERROR mismatched types
        //~^ SUGGESTION ;
    };
}

fn dont_suggest_breaking_from_for_nested_in_loop() {
    let _ = loop {
        for _ in 0.. {
            returns_i32() //~ ERROR mismatched types
            //~^ SUGGESTION ;
        }
    };
}

fn main() {}
