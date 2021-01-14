#![feature(capture_disjoint_fields)]
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

    let c = #[rustc_capture_analysis] || {
        let s2 = S {
            a: format!("New a"),
            ..s
        };
    };

    c();
}
