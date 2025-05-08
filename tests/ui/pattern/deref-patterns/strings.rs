//@ run-pass
//! Test deref patterns using string and bytestring literals.

#![feature(deref_patterns)]
#![allow(incomplete_features)]

fn main() {
    for (test_in, test_expect) in [("zero", 0), ("one", 1), ("two", 2)] {
        // Test string literal patterns having type `str`.
        let test_actual = match *test_in {
            "zero" => 0,
            "one" => 1,
            _ => 2,
        };
        assert_eq!(test_actual, test_expect);

        // Test matching on `&mut str`.
        let test_actual = match &mut *test_in.to_string() {
            "zero" => 0,
            "one" => 1,
            _ => 2,
        };
        assert_eq!(test_actual, test_expect);

        // Test string literals in deref patterns.
        let test_actual = match test_in.to_string() {
            deref!("zero") => 0,
            "one" => 1,
            _ => 2,
        };
        assert_eq!(test_actual, test_expect);

        // Test peeling references in addition to smart pointers.
        let test_actual = match &test_in.to_string() {
            deref!("zero") => 0,
            "one" => 1,
            _ => 2,
        };
        assert_eq!(test_actual, test_expect);
    }

    // Test that we can still mutate in the match arm after using a literal to test equality:
    let mut test = "test".to_string();
    if let deref!(s @ "test") = &mut test {
        s.make_ascii_uppercase();
    }
    assert_eq!(test, "TEST");

    for (test_in, test_expect) in [(b"0", 0), (b"1", 1), (b"2", 2)] {
        // Test byte string literal patterns having type `[u8; N]`
        let test_actual = match *test_in {
            b"0" => 0,
            b"1" => 1,
            _ => 2,
        };
        assert_eq!(test_actual, test_expect);

        // Test byte string literal patterns having type `[u8]`
        let test_actual = match *(test_in as &[u8]) {
            b"0" => 0,
            b"1" => 1,
            _ => 2,
        };
        assert_eq!(test_actual, test_expect);

        // Test matching on `&mut [u8; N]`.
        let test_actual = match &mut test_in.clone() {
            b"0" => 0,
            b"1" => 1,
            _ => 2,
        };
        assert_eq!(test_actual, test_expect);

        // Test matching on `&mut [u8]`.
        let test_actual = match &mut test_in.clone()[..] {
            b"0" => 0,
            b"1" => 1,
            _ => 2,
        };
        assert_eq!(test_actual, test_expect);

        // Test byte string literals used as arrays in deref patterns.
        let test_actual = match Box::new(*test_in) {
            deref!(b"0") => 0,
            b"1" => 1,
            _ => 2,
        };
        assert_eq!(test_actual, test_expect);

        // Test byte string literals used as slices in deref patterns.
        let test_actual = match test_in.to_vec() {
            deref!(b"0") => 0,
            b"1" => 1,
            _ => 2,
        };
        assert_eq!(test_actual, test_expect);
    }
}
