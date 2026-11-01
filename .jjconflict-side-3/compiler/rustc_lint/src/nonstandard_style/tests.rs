use super::{is_upper_camel_case, to_upper_camel_case};

#[test]
fn camel_case() {
    assert!(!is_upper_camel_case("userData"));
    assert_eq!(to_upper_camel_case("userData"), "UserData");

    assert!(is_upper_camel_case("X86_64"));

    assert!(!is_upper_camel_case("X86__64"));
    assert_eq!(to_upper_camel_case("X86__64"), "X86_64");

    assert!(!is_upper_camel_case("Abc_123"));
    assert_eq!(to_upper_camel_case("Abc_123"), "Abc123");

    assert!(!is_upper_camel_case("A1_b2_c3"));
    assert_eq!(to_upper_camel_case("A1_b2_c3"), "A1B2C3");

    assert!(!is_upper_camel_case("ONE_TWO_THREE"));
    assert_eq!(to_upper_camel_case("ONE_TWO_THREE"), "OneTwoThree");

    // FIXME(@Jules-Bertholet): This test doesn't work due to what I believe
    // is a Unicode spec bug - uppercase Georgian letters have
    // incorrect titlecase mappings.
    // I've reported it to Unicode.
    // Georgian mtavruli is only used in all-caps
    //assert!(!is_upper_camel_case("ᲫალაᲔრთობაშია"));
    //assert_eq!(to_upper_camel_case("ᲫალაᲔრთობაშია"), "ძალა_ერთობაშია");

    assert!(!is_upper_camel_case("ǇǊaaaǄooo"));
    assert_eq!(to_upper_camel_case("ǇǊaaǈǊaǄooo"), "ǈǌaaǈǋaǅooo");

    // Final sigma
    assert!(!is_upper_camel_case("ΦΙΛΟΣ_ΦΙΛΟΣ"));
    assert_eq!(to_upper_camel_case("ΦΙΛΟΣ_ΦΙΛΟΣ"), "ΦιλοςΦιλος");
    assert!(is_upper_camel_case("ΦιλοσΦιλοσ"));
}
