use rustc_apfloat::ieee::Double;
use rustc_apfloat::ParseError;

#[test]
fn string_decimal_death() {
    assert_eq!("".parse::<Double>(), Err(ParseError("Invalid string length")));
    assert_eq!("+".parse::<Double>(), Err(ParseError("String has no digits")));
    assert_eq!("-".parse::<Double>(), Err(ParseError("String has no digits")));

    assert_eq!("\0".parse::<Double>(), Err(ParseError("Invalid character in significand")));
    assert_eq!("1\0".parse::<Double>(), Err(ParseError("Invalid character in significand")));
    assert_eq!("1\02".parse::<Double>(), Err(ParseError("Invalid character in significand")));
    assert_eq!("1\02e1".parse::<Double>(), Err(ParseError("Invalid character in significand")));
    assert_eq!("1e\0".parse::<Double>(), Err(ParseError("Invalid character in exponent")));
    assert_eq!("1e1\0".parse::<Double>(), Err(ParseError("Invalid character in exponent")));
    assert_eq!("1e1\02".parse::<Double>(), Err(ParseError("Invalid character in exponent")));

    assert_eq!("1.0f".parse::<Double>(), Err(ParseError("Invalid character in significand")));

    assert_eq!("..".parse::<Double>(), Err(ParseError("String contains multiple dots")));
    assert_eq!("..0".parse::<Double>(), Err(ParseError("String contains multiple dots")));
    assert_eq!("1.0.0".parse::<Double>(), Err(ParseError("String contains multiple dots")));
}

#[test]
fn string_decimal_significand_death() {
    assert_eq!(".".parse::<Double>(), Err(ParseError("Significand has no digits")));
    assert_eq!("+.".parse::<Double>(), Err(ParseError("Significand has no digits")));
    assert_eq!("-.".parse::<Double>(), Err(ParseError("Significand has no digits")));

    assert_eq!("e".parse::<Double>(), Err(ParseError("Significand has no digits")));
    assert_eq!("+e".parse::<Double>(), Err(ParseError("Significand has no digits")));
    assert_eq!("-e".parse::<Double>(), Err(ParseError("Significand has no digits")));

    assert_eq!("e1".parse::<Double>(), Err(ParseError("Significand has no digits")));
    assert_eq!("+e1".parse::<Double>(), Err(ParseError("Significand has no digits")));
    assert_eq!("-e1".parse::<Double>(), Err(ParseError("Significand has no digits")));

    assert_eq!(".e1".parse::<Double>(), Err(ParseError("Significand has no digits")));
    assert_eq!("+.e1".parse::<Double>(), Err(ParseError("Significand has no digits")));
    assert_eq!("-.e1".parse::<Double>(), Err(ParseError("Significand has no digits")));

    assert_eq!(".e".parse::<Double>(), Err(ParseError("Significand has no digits")));
    assert_eq!("+.e".parse::<Double>(), Err(ParseError("Significand has no digits")));
    assert_eq!("-.e".parse::<Double>(), Err(ParseError("Significand has no digits")));
}

#[test]
fn string_decimal_exponent_death() {
    assert_eq!("1e".parse::<Double>(), Err(ParseError("Exponent has no digits")));
    assert_eq!("+1e".parse::<Double>(), Err(ParseError("Exponent has no digits")));
    assert_eq!("-1e".parse::<Double>(), Err(ParseError("Exponent has no digits")));

    assert_eq!("1.e".parse::<Double>(), Err(ParseError("Exponent has no digits")));
    assert_eq!("+1.e".parse::<Double>(), Err(ParseError("Exponent has no digits")));
    assert_eq!("-1.e".parse::<Double>(), Err(ParseError("Exponent has no digits")));

    assert_eq!(".1e".parse::<Double>(), Err(ParseError("Exponent has no digits")));
    assert_eq!("+.1e".parse::<Double>(), Err(ParseError("Exponent has no digits")));
    assert_eq!("-.1e".parse::<Double>(), Err(ParseError("Exponent has no digits")));

    assert_eq!("1.1e".parse::<Double>(), Err(ParseError("Exponent has no digits")));
    assert_eq!("+1.1e".parse::<Double>(), Err(ParseError("Exponent has no digits")));
    assert_eq!("-1.1e".parse::<Double>(), Err(ParseError("Exponent has no digits")));

    assert_eq!("1e+".parse::<Double>(), Err(ParseError("Exponent has no digits")));
    assert_eq!("1e-".parse::<Double>(), Err(ParseError("Exponent has no digits")));

    assert_eq!(".1e".parse::<Double>(), Err(ParseError("Exponent has no digits")));
    assert_eq!(".1e+".parse::<Double>(), Err(ParseError("Exponent has no digits")));
    assert_eq!(".1e-".parse::<Double>(), Err(ParseError("Exponent has no digits")));

    assert_eq!("1.0e".parse::<Double>(), Err(ParseError("Exponent has no digits")));
    assert_eq!("1.0e+".parse::<Double>(), Err(ParseError("Exponent has no digits")));
    assert_eq!("1.0e-".parse::<Double>(), Err(ParseError("Exponent has no digits")));
}

#[test]
fn string_hexadecimal_death() {
    assert_eq!("0x".parse::<Double>(), Err(ParseError("Invalid string")));
    assert_eq!("+0x".parse::<Double>(), Err(ParseError("Invalid string")));
    assert_eq!("-0x".parse::<Double>(), Err(ParseError("Invalid string")));

    assert_eq!("0x0".parse::<Double>(), Err(ParseError("Hex strings require an exponent")));
    assert_eq!("+0x0".parse::<Double>(), Err(ParseError("Hex strings require an exponent")));
    assert_eq!("-0x0".parse::<Double>(), Err(ParseError("Hex strings require an exponent")));

    assert_eq!("0x0.".parse::<Double>(), Err(ParseError("Hex strings require an exponent")));
    assert_eq!("+0x0.".parse::<Double>(), Err(ParseError("Hex strings require an exponent")));
    assert_eq!("-0x0.".parse::<Double>(), Err(ParseError("Hex strings require an exponent")));

    assert_eq!("0x.0".parse::<Double>(), Err(ParseError("Hex strings require an exponent")));
    assert_eq!("+0x.0".parse::<Double>(), Err(ParseError("Hex strings require an exponent")));
    assert_eq!("-0x.0".parse::<Double>(), Err(ParseError("Hex strings require an exponent")));

    assert_eq!("0x0.0".parse::<Double>(), Err(ParseError("Hex strings require an exponent")));
    assert_eq!("+0x0.0".parse::<Double>(), Err(ParseError("Hex strings require an exponent")));
    assert_eq!("-0x0.0".parse::<Double>(), Err(ParseError("Hex strings require an exponent")));

    assert_eq!("0x\0".parse::<Double>(), Err(ParseError("Invalid character in significand")));
    assert_eq!("0x1\0".parse::<Double>(), Err(ParseError("Invalid character in significand")));
    assert_eq!("0x1\02".parse::<Double>(), Err(ParseError("Invalid character in significand")));
    assert_eq!("0x1\02p1".parse::<Double>(), Err(ParseError("Invalid character in significand")));
    assert_eq!("0x1p\0".parse::<Double>(), Err(ParseError("Invalid character in exponent")));
    assert_eq!("0x1p1\0".parse::<Double>(), Err(ParseError("Invalid character in exponent")));
    assert_eq!("0x1p1\02".parse::<Double>(), Err(ParseError("Invalid character in exponent")));

    assert_eq!("0x1p0f".parse::<Double>(), Err(ParseError("Invalid character in exponent")));

    assert_eq!("0x..p1".parse::<Double>(), Err(ParseError("String contains multiple dots")));
    assert_eq!("0x..0p1".parse::<Double>(), Err(ParseError("String contains multiple dots")));
    assert_eq!("0x1.0.0p1".parse::<Double>(), Err(ParseError("String contains multiple dots")));
}

#[test]
fn string_hexadecimal_significand_death() {
    assert_eq!("0x.".parse::<Double>(), Err(ParseError("Significand has no digits")));
    assert_eq!("+0x.".parse::<Double>(), Err(ParseError("Significand has no digits")));
    assert_eq!("-0x.".parse::<Double>(), Err(ParseError("Significand has no digits")));

    assert_eq!("0xp".parse::<Double>(), Err(ParseError("Significand has no digits")));
    assert_eq!("+0xp".parse::<Double>(), Err(ParseError("Significand has no digits")));
    assert_eq!("-0xp".parse::<Double>(), Err(ParseError("Significand has no digits")));

    assert_eq!("0xp+".parse::<Double>(), Err(ParseError("Significand has no digits")));
    assert_eq!("+0xp+".parse::<Double>(), Err(ParseError("Significand has no digits")));
    assert_eq!("-0xp+".parse::<Double>(), Err(ParseError("Significand has no digits")));

    assert_eq!("0xp-".parse::<Double>(), Err(ParseError("Significand has no digits")));
    assert_eq!("+0xp-".parse::<Double>(), Err(ParseError("Significand has no digits")));
    assert_eq!("-0xp-".parse::<Double>(), Err(ParseError("Significand has no digits")));

    assert_eq!("0x.p".parse::<Double>(), Err(ParseError("Significand has no digits")));
    assert_eq!("+0x.p".parse::<Double>(), Err(ParseError("Significand has no digits")));
    assert_eq!("-0x.p".parse::<Double>(), Err(ParseError("Significand has no digits")));

    assert_eq!("0x.p+".parse::<Double>(), Err(ParseError("Significand has no digits")));
    assert_eq!("+0x.p+".parse::<Double>(), Err(ParseError("Significand has no digits")));
    assert_eq!("-0x.p+".parse::<Double>(), Err(ParseError("Significand has no digits")));

    assert_eq!("0x.p-".parse::<Double>(), Err(ParseError("Significand has no digits")));
    assert_eq!("+0x.p-".parse::<Double>(), Err(ParseError("Significand has no digits")));
    assert_eq!("-0x.p-".parse::<Double>(), Err(ParseError("Significand has no digits")));
}

#[test]
fn string_hexadecimal_exponent_death() {
    assert_eq!("0x1p".parse::<Double>(), Err(ParseError("Exponent has no digits")));
    assert_eq!("+0x1p".parse::<Double>(), Err(ParseError("Exponent has no digits")));
    assert_eq!("-0x1p".parse::<Double>(), Err(ParseError("Exponent has no digits")));

    assert_eq!("0x1p+".parse::<Double>(), Err(ParseError("Exponent has no digits")));
    assert_eq!("+0x1p+".parse::<Double>(), Err(ParseError("Exponent has no digits")));
    assert_eq!("-0x1p+".parse::<Double>(), Err(ParseError("Exponent has no digits")));

    assert_eq!("0x1p-".parse::<Double>(), Err(ParseError("Exponent has no digits")));
    assert_eq!("+0x1p-".parse::<Double>(), Err(ParseError("Exponent has no digits")));
    assert_eq!("-0x1p-".parse::<Double>(), Err(ParseError("Exponent has no digits")));

    assert_eq!("0x1.p".parse::<Double>(), Err(ParseError("Exponent has no digits")));
    assert_eq!("+0x1.p".parse::<Double>(), Err(ParseError("Exponent has no digits")));
    assert_eq!("-0x1.p".parse::<Double>(), Err(ParseError("Exponent has no digits")));

    assert_eq!("0x1.p+".parse::<Double>(), Err(ParseError("Exponent has no digits")));
    assert_eq!("+0x1.p+".parse::<Double>(), Err(ParseError("Exponent has no digits")));
    assert_eq!("-0x1.p+".parse::<Double>(), Err(ParseError("Exponent has no digits")));

    assert_eq!("0x1.p-".parse::<Double>(), Err(ParseError("Exponent has no digits")));
    assert_eq!("+0x1.p-".parse::<Double>(), Err(ParseError("Exponent has no digits")));
    assert_eq!("-0x1.p-".parse::<Double>(), Err(ParseError("Exponent has no digits")));

    assert_eq!("0x.1p".parse::<Double>(), Err(ParseError("Exponent has no digits")));
    assert_eq!("+0x.1p".parse::<Double>(), Err(ParseError("Exponent has no digits")));
    assert_eq!("-0x.1p".parse::<Double>(), Err(ParseError("Exponent has no digits")));

    assert_eq!("0x.1p+".parse::<Double>(), Err(ParseError("Exponent has no digits")));
    assert_eq!("+0x.1p+".parse::<Double>(), Err(ParseError("Exponent has no digits")));
    assert_eq!("-0x.1p+".parse::<Double>(), Err(ParseError("Exponent has no digits")));

    assert_eq!("0x.1p-".parse::<Double>(), Err(ParseError("Exponent has no digits")));
    assert_eq!("+0x.1p-".parse::<Double>(), Err(ParseError("Exponent has no digits")));
    assert_eq!("-0x.1p-".parse::<Double>(), Err(ParseError("Exponent has no digits")));

    assert_eq!("0x1.1p".parse::<Double>(), Err(ParseError("Exponent has no digits")));
    assert_eq!("+0x1.1p".parse::<Double>(), Err(ParseError("Exponent has no digits")));
    assert_eq!("-0x1.1p".parse::<Double>(), Err(ParseError("Exponent has no digits")));

    assert_eq!("0x1.1p+".parse::<Double>(), Err(ParseError("Exponent has no digits")));
    assert_eq!("+0x1.1p+".parse::<Double>(), Err(ParseError("Exponent has no digits")));
    assert_eq!("-0x1.1p+".parse::<Double>(), Err(ParseError("Exponent has no digits")));

    assert_eq!("0x1.1p-".parse::<Double>(), Err(ParseError("Exponent has no digits")));
    assert_eq!("+0x1.1p-".parse::<Double>(), Err(ParseError("Exponent has no digits")));
    assert_eq!("-0x1.1p-".parse::<Double>(), Err(ParseError("Exponent has no digits")));
}
