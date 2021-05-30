use proc_macro::Literal;

pub fn test() {
    test_parse_literal();
}

fn test_parse_literal() {
    assert_eq!("1".parse::<Literal>().unwrap().to_string(), "1");
    assert_eq!("1.0".parse::<Literal>().unwrap().to_string(), "1.0");
    assert_eq!("'a'".parse::<Literal>().unwrap().to_string(), "'a'");
    assert_eq!("\"\n\"".parse::<Literal>().unwrap().to_string(), "\"\n\"");
    assert_eq!("b\"\"".parse::<Literal>().unwrap().to_string(), "b\"\"");
    assert_eq!("r##\"\"##".parse::<Literal>().unwrap().to_string(), "r##\"\"##");
    assert_eq!("10ulong".parse::<Literal>().unwrap().to_string(), "10ulong");

    assert!("0 1".parse::<Literal>().is_err());
    assert!("'a".parse::<Literal>().is_err());
    assert!(" 0".parse::<Literal>().is_err());
    assert!("0 ".parse::<Literal>().is_err());
    assert!("/* comment */0".parse::<Literal>().is_err());
    assert!("0/* comment */".parse::<Literal>().is_err());
    assert!("0// comment".parse::<Literal>().is_err());
}
