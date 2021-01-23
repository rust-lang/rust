// run-pass

// Test that functional record update/struct update syntax works inside
// a closure when the feature `capture_disjoint_fields` is enabled.

#![feature(capture_disjoint_fields)]
//~^ WARNING: the feature `capture_disjoint_fields` is incomplete
//~| NOTE: `#[warn(incomplete_features)]` on by default
//~| NOTE: see issue #53488 <https://github.com/rust-lang/rust/issues/53488>

struct S {
    a: String,
    b: String,
}

fn main() {
    let a = String::new();
    let b = String::new();
    let s = S {a, b};

    let c = || {
        let s2 = S {
            a: format!("New a"),
            ..s
        };
        println!("{} {}", s2.a, s2.b);
    };

    c();
}
