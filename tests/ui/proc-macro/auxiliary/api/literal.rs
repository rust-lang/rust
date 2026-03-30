// ignore-tidy-linelength

#![feature(f16)]

use proc_macro::Literal;

pub fn test() {
    test_display_literal();
    test_parse_literal();
    test_literal_value();
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

    assert_eq!(Literal::string("aA").to_string(), r#"  "aA"  "#.trim());
    assert_eq!(Literal::string("\t").to_string(), r#"  "\t"  "#.trim());
    assert_eq!(Literal::string("❤").to_string(), r#"  "❤"  "#.trim());
    assert_eq!(Literal::string("'").to_string(), r#"  "'"  "#.trim());
    assert_eq!(Literal::string("\"").to_string(), r#"  "\""  "#.trim());
    assert_eq!(Literal::string("\0").to_string(), r#"  "\0"  "#.trim());
    assert_eq!(Literal::string("\u{1}").to_string(), r#"  "\u{1}"  "#.trim());

    assert_eq!(Literal::byte_string(b"aA").to_string(), r#"  b"aA"  "#.trim());
    assert_eq!(Literal::byte_string(b"\t").to_string(), r#"  b"\t"  "#.trim());
    assert_eq!(Literal::byte_string(b"'").to_string(), r#"  b"'"  "#.trim());
    assert_eq!(Literal::byte_string(b"\"").to_string(), r#"  b"\""  "#.trim());
    assert_eq!(Literal::byte_string(b"\0").to_string(), r#"  b"\0"  "#.trim());
    assert_eq!(Literal::byte_string(b"\x01").to_string(), r#"  b"\x01"  "#.trim());

    assert_eq!(Literal::c_string(c"aA").to_string(), r#"  c"aA"  "#.trim());
    assert_eq!(Literal::c_string(c"\t").to_string(), r#"  c"\t"  "#.trim());
    assert_eq!(Literal::c_string(c"❤").to_string(), r#"  c"❤"  "#.trim());
    assert_eq!(Literal::c_string(c"\'").to_string(), r#"  c"'"  "#.trim());
    assert_eq!(Literal::c_string(c"\"").to_string(), r#"  c"\""  "#.trim());
    assert_eq!(Literal::c_string(c"\x7f\xff\xfe\u{333}").to_string(), r#"  c"\u{7f}\xff\xfe\u{333}"  "#.trim());

    assert_eq!(Literal::character('a').to_string(), r#"  'a'  "#.trim());
    assert_eq!(Literal::character('\t').to_string(), r#"  '\t'  "#.trim());
    assert_eq!(Literal::character('❤').to_string(), r#"  '❤'  "#.trim());
    assert_eq!(Literal::character('\'').to_string(), r#"  '\''  "#.trim());
    assert_eq!(Literal::character('"').to_string(), r#"  '"'  "#.trim());
    assert_eq!(Literal::character('\0').to_string(), r#"  '\0'  "#.trim());
    assert_eq!(Literal::character('\u{1}').to_string(), r#"  '\u{1}'  "#.trim());

    assert_eq!(Literal::byte_character(b'a').to_string(), r#"  b'a'  "#.trim());
    assert_eq!(Literal::byte_character(b'\t').to_string(), r#"  b'\t'  "#.trim());
    assert_eq!(Literal::byte_character(b'\'').to_string(), r#"  b'\''  "#.trim());
    assert_eq!(Literal::byte_character(b'"').to_string(), r#"  b'"'  "#.trim());
    assert_eq!(Literal::byte_character(0).to_string(), r#"  b'\0'  "#.trim());
    assert_eq!(Literal::byte_character(1).to_string(), r#"  b'\x01'  "#.trim());
}

fn test_parse_literal() {
    assert_eq!("1".parse::<Literal>().unwrap().to_string(), "1");
    assert_eq!("1.0".parse::<Literal>().unwrap().to_string(), "1.0");
    assert_eq!("'a'".parse::<Literal>().unwrap().to_string(), "'a'");
    assert_eq!("b'a'".parse::<Literal>().unwrap().to_string(), "b'a'");
    assert_eq!("\"\n\"".parse::<Literal>().unwrap().to_string(), "\"\n\"");
    assert_eq!("b\"\"".parse::<Literal>().unwrap().to_string(), "b\"\"");
    assert_eq!("c\"\"".parse::<Literal>().unwrap().to_string(), "c\"\"");
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

fn test_literal_value() {
    assert!(Literal::u32_suffixed(10).u8_value().is_err());
    assert!(Literal::u32_suffixed(11).f32_value().is_err());
    assert!(Literal::u32_suffixed(12).u64_value().is_err());
    assert_eq!(Literal::u32_suffixed(13).u32_value(), Ok(13u32));
    assert_eq!(Literal::u32_unsuffixed(14).u64_value(), Ok(14u64));
    assert_eq!(Literal::u32_unsuffixed(15).u8_value(), Ok(15u8));
    assert!(Literal::u32_unsuffixed(400).u8_value().is_err());
    assert_eq!(Literal::u32_unsuffixed(401).u16_value(), Ok(401u16));
    assert!(Literal::i32_unsuffixed(-402).u16_value().is_err());
    assert_eq!(Literal::i32_unsuffixed(-403).i16_value(), Ok(-403i16));
    assert_eq!(Literal::u32_unsuffixed(0xff).u16_value(), Ok(0xffu16));
    assert_eq!(Literal::u32_unsuffixed(0b11).u16_value(), Ok(0b11u16));
    assert_eq!(Literal::u32_unsuffixed(0o11).u16_value(), Ok(0o11u16));

    assert!(Literal::f32_suffixed(9.).u8_value().is_err());
    assert!(Literal::f32_suffixed(10.).f64_value().is_err());
    assert!(Literal::f32_suffixed(11.).f16_value().is_err());
    assert_eq!(Literal::f32_suffixed(12.).f32_value().map(|f| f.to_string()), Ok("12".to_string()));
    assert_eq!(Literal::f32_unsuffixed(13.).f32_value().map(|f| f.to_string()), Ok("13".to_string()));
    assert_eq!(Literal::f32_unsuffixed(14.).f64_value().map(|f| f.to_string()), Ok("14".to_string()));
    assert_eq!(Literal::f32_unsuffixed(15.).f16_value().map(|f| f.to_string()), Ok("15".to_string()));
}
