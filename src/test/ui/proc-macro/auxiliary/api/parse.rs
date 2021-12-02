// ignore-tidy-linelength

use proc_macro::Literal;

pub fn test() {
    test_display_literal();
    test_parse_literal();
}

fn test_display_literal() {
    assert_eq!(Literal::isize_unsuffixed(-10).to_string(), "-10");
    assert_eq!(Literal::isize_suffixed(-10).to_string(), "-10isize");
    assert_eq!(Literal::f32_unsuffixed(-10.0).to_string(), "-10.0");
    assert_eq!(Literal::f32_suffixed(-10.0).to_string(), "-10f32");
    assert_eq!(Literal::f64_unsuffixed(-10.0).to_string(), "-10.0");
    assert_eq!(Literal::f64_suffixed(-10.0).to_string(), "-10f64");
    assert_eq!(
        Literal::f64_unsuffixed(1e100).to_string(),
        "10000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000.0",
    );
}

fn test_parse_literal() {
    assert_eq!("1".parse::<Literal>().unwrap().to_string(), "1");
    assert_eq!("1.0".parse::<Literal>().unwrap().to_string(), "1.0");
    assert_eq!("'a'".parse::<Literal>().unwrap().to_string(), "'a'");
    assert_eq!("\"\n\"".parse::<Literal>().unwrap().to_string(), "\"\n\"");
    assert_eq!("b\"\"".parse::<Literal>().unwrap().to_string(), "b\"\"");
    assert_eq!("r##\"\"##".parse::<Literal>().unwrap().to_string(), "r##\"\"##");
    assert_eq!("10ulong".parse::<Literal>().unwrap().to_string(), "10ulong");
    assert_eq!("-10ulong".parse::<Literal>().unwrap().to_string(), "-10ulong");

    assert!("true".parse::<Literal>().is_err());
    assert!(".8".parse::<Literal>().is_err());
    assert!("0 1".parse::<Literal>().is_err());
    assert!("'a".parse::<Literal>().is_err());
    assert!(" 0".parse::<Literal>().is_err());
    assert!("0 ".parse::<Literal>().is_err());
    assert!("/* comment */0".parse::<Literal>().is_err());
    assert!("0/* comment */".parse::<Literal>().is_err());
    assert!("0// comment".parse::<Literal>().is_err());
    assert!("- 10".parse::<Literal>().is_err());
    assert!("-'x'".parse::<Literal>().is_err());
}
