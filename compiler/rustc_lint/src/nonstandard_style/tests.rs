use super::{is_camel_case, to_camel_case};

#[test]
fn camel_case() {
    assert!(!is_camel_case("userData"));
    assert_eq!(to_camel_case("userData"), "UserData");

    assert!(is_camel_case("X86_64"));

    assert!(!is_camel_case("X86__64"));
    assert_eq!(to_camel_case("X86__64"), "X86_64");

    assert!(!is_camel_case("Abc_123"));
    assert_eq!(to_camel_case("Abc_123"), "Abc123");

    assert!(!is_camel_case("A1_b2_c3"));
    assert_eq!(to_camel_case("A1_b2_c3"), "A1B2C3");

    assert!(!is_camel_case("ONE_TWO_THREE"));
    assert_eq!(to_camel_case("ONE_TWO_THREE"), "OneTwoThree");
}
