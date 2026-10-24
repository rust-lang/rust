//@ check-pass
// regression test for https://github.com/rust-lang/rust/issues/53485#issuecomment-885393452

struct A {
    name: String,
}

fn main() {
    let a = &[
        A {
            name: "1".to_string(),
        },
        A {
            name: "2".to_string(),
        },
    ];
    assert!(a.is_sorted_by_key(|a| a.name.as_str()));
}
