pub use std::cfg_if;

cfg_if! {
    if #[cfg(test)] {
        fn foo() -> bool { true }
    } else {
        fn foo() -> bool { false }
    }
}

#[test]
fn cfg_if_test {
    assert!(foo());
}
