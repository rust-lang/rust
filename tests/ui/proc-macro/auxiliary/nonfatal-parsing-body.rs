use std::fmt::Debug;
use std::panic::catch_unwind;
use std::str::FromStr;

use proc_macro::*;

use self::Mode::*;

// FIXME: all cases should become `NormalOk` or `NormalErr`
#[derive(PartialEq, Clone, Copy)]
enum Mode {
    NormalOk,
    NormalErr,
    OtherError,
    OtherWithPanic,
}

fn parse<T>(s: &str, mode: Mode)
where
    T: FromStr<Err = LexError> + Debug,
{
    match mode {
        NormalOk => {
            let t = T::from_str(s);
            println!("{:?}", t);
            assert!(t.is_ok());
        }
        NormalErr => {
            let t = T::from_str(s);
            println!("{:?}", t);
            assert!(t.is_err());
        }
        OtherError => {
            println!("{:?}", T::from_str(s));
        }
        OtherWithPanic => {
            if catch_unwind(|| println!("{:?}", T::from_str(s))).is_ok() {
                eprintln!("{s} did not panic");
            }
        }
    }
}

fn stream(s: &str, mode: Mode) {
    parse::<TokenStream>(s, mode);
}

fn lit(s: &str, mode: Mode) {
    parse::<Literal>(s, mode);
    if mode == NormalOk {
        let Ok(lit) = Literal::from_str(s) else {
            panic!("literal was not ok");
        };
        let Ok(stream) = TokenStream::from_str(s) else {
            panic!("tokenstream was not ok, but literal was");
        };
        let Some(tree) = stream.into_iter().next() else {
            panic!("tokenstream should have a tokentree");
        };
        if let TokenTree::Literal(tokenstream_lit) = tree {
            assert_eq!(lit.to_string(), tokenstream_lit.to_string());
        }
    }
}

pub fn run() {
    // returns Ok(valid instance)
    lit("123", NormalOk);
    lit("\"ab\"", NormalOk);
    lit("\'b\'", NormalOk);
    lit("'b'", NormalOk);
    lit("b\"b\"", NormalOk);
    lit("c\"b\"", NormalOk);
    lit("cr\"b\"", NormalOk);
    lit("b'b'", NormalOk);
    lit("256u8", NormalOk);
    lit("-256u8", NormalOk);
    stream("-256u8", NormalOk);
    lit("0b11111000000001111i16", NormalOk);
    lit("0xf32", NormalOk);
    lit("0b0f32", NormalOk);
    lit("2E4", NormalOk);
    lit("2.2E-4f64", NormalOk);
    lit("18u8E", NormalOk);
    lit("18.0u8E", NormalOk);
    lit("cr#\"// /* // \n */\"#", NormalOk);
    lit("'\\''", NormalOk);
    lit("'\\\''", NormalOk);
    lit(&format!("r{0}\"a\"{0}", "#".repeat(255)), NormalOk);
    stream("fn main() { println!(\"Hello, world!\") }", NormalOk);
    stream("18.u8E", NormalOk);
    stream("18.0f32", NormalOk);
    stream("18.0f34", NormalOk);
    stream("18.bu8", NormalOk);
    stream("3//\n4", NormalOk);
    stream(
        "\'c\'/*\n
    */",
        NormalOk,
    );
    stream("/*a*/ //", NormalOk);

    println!("### ERRORS");

    // returns Err(LexError)
    lit("\'c\'/**/", NormalErr);
    lit(" 0", NormalErr);
    lit("0 ", NormalErr);
    lit("0//", NormalErr);
    lit("3//\n4", NormalErr);
    lit("18.u8E", NormalErr);
    lit("/*a*/ //", NormalErr);
    // FIXME: all of the cases below should return an Err and emit no diagnostics, but don't yet.

    // emits diagnostics and returns LexError
    lit("r'r'", OtherError);
    lit("c'r'", OtherError);

    // emits diagnostic and returns a seemingly valid tokenstream
    stream("r'r'", OtherError);
    stream("c'r'", OtherError);

    for parse in [stream as fn(&str, Mode), lit] {
        // emits diagnostic(s), then panics
        parse("1 ) 2", OtherWithPanic);
        parse("( x  [ ) ]", OtherWithPanic);
        parse("r#", OtherWithPanic);

        // emits diagnostic(s), then returns Ok(Literal { kind: ErrWithGuar, .. })
        parse("0b2", OtherError);
        parse("0bf32", OtherError);
        parse("0b0.0f32", OtherError);
        parse("'\''", OtherError);
        parse(
            "'
'", OtherError,
        );
        parse(&format!("r{0}\"a\"{0}", "#".repeat(256)), OtherWithPanic);

        // emits diagnostic, then, when parsing as a lit, returns LexError, otherwise ErrWithGuar
        parse("/*a*/ 0b2 //", OtherError);
    }
}
