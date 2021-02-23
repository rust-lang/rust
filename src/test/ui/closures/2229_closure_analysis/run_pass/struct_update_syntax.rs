//check-pass
#![feature(capture_disjoint_fields)]
//~^ WARNING: the feature `capture_disjoint_fields` is incomplete
#![feature(rustc_attrs)]

struct S {
    a: String,
    b: String,
}

fn main() {
    let s = S {
        a: String::new(),
        b: String::new(),
    };

    let c = || {
        let s2 = S {
            a: format!("New a"),
            ..s
        };
    };

    c();
}
