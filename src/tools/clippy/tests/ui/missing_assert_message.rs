#![allow(unused)]
#![warn(clippy::missing_assert_message)]

macro_rules! bar {
    ($( $x:expr ),*) => {
        foo()
    };
}

fn main() {}

// Should trigger warning
fn asserts_without_message() {
    assert!(foo());
    assert_eq!(foo(), foo());
    assert_ne!(foo(), foo());
    debug_assert!(foo());
    debug_assert_eq!(foo(), foo());
    debug_assert_ne!(foo(), foo());
}

// Should trigger warning
fn asserts_without_message_but_with_macro_calls() {
    assert!(bar!(true));
    assert!(bar!(true, false));
    assert_eq!(bar!(true), foo());
    assert_ne!(bar!(true, true), bar!(true));
}

// Should trigger warning
fn asserts_with_trailing_commas() {
    assert!(foo(),);
    assert_eq!(foo(), foo(),);
    assert_ne!(foo(), foo(),);
    debug_assert!(foo(),);
    debug_assert_eq!(foo(), foo(),);
    debug_assert_ne!(foo(), foo(),);
}

// Should not trigger warning
fn asserts_with_message_and_with_macro_calls() {
    assert!(bar!(true), "msg");
    assert!(bar!(true, false), "msg");
    assert_eq!(bar!(true), foo(), "msg");
    assert_ne!(bar!(true, true), bar!(true), "msg");
}

// Should not trigger warning
fn asserts_with_message() {
    assert!(foo(), "msg");
    assert_eq!(foo(), foo(), "msg");
    assert_ne!(foo(), foo(), "msg");
    debug_assert!(foo(), "msg");
    debug_assert_eq!(foo(), foo(), "msg");
    debug_assert_ne!(foo(), foo(), "msg");
}

// Should not trigger warning
#[test]
fn asserts_without_message_but_inside_a_test_function() {
    assert!(foo());
    assert_eq!(foo(), foo());
    assert_ne!(foo(), foo());
    debug_assert!(foo());
    debug_assert_eq!(foo(), foo());
    debug_assert_ne!(foo(), foo());
}

// Should not trigger warning
#[cfg(test)]
mod tests {
    fn asserts_without_message_but_inside_a_test_module() {
        assert!(foo());
        assert_eq!(foo(), foo());
        assert_ne!(foo(), foo());
        debug_assert!(foo());
        debug_assert_eq!(foo(), foo());
        debug_assert_ne!(foo(), foo());
    }
}

fn foo() -> bool {
    true
}
