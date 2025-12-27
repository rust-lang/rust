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

fn print_unspanned<T>(s: &str) where T: FromStr<Err = LexError> + Debug {
    let t = T::from_str(s);
    let mut s = format!("{t:?}");
    while let Some((l, r)) = s.split_once("span: #") {
        let (_, r) = r.split_once(")").unwrap();
        s = format!("{l}span: Span{r}");
    }
    println!("{s}");
}

fn parse<T>(s: &str, mode: Mode)
where
    T: FromStr<Err = LexError> + Debug,
{
    match mode {
        NormalOk => {
            print_unspanned::<T>(s);
            //assert!(T::from_str(s).is_ok());
        }
        NormalErr => {
            print_unspanned::<T>(s);
            //assert!(T::from_str(s).is_err());
        }
        OtherError => {
            print_unspanned::<T>(s);
        }
        OtherWithPanic => {
            if catch_unwind(|| print_unspanned::<T>(s)).is_ok() {
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
    /*if mode == NormalOk {
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
    }*/
}

pub fn run() {
    assert_eq!("\'", "'");
    // returns Ok(valid instance)
    lit("r\"g\"", NormalOk);
    lit("r#\"g\"#", NormalOk);
    lit("123", NormalOk);
    lit("\"ab\"", NormalOk);
    lit("\'b\'", NormalOk);
    lit("'b'", NormalOk);
    lit("b\"b\"", NormalOk);
    lit("c\"b\"", NormalOk);
    lit("cr\"b\"", NormalOk);
    lit("'\\''", NormalOk);
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
    lit("\"\"", NormalOk);

    println!("### ERRORS");

    // returns Err(LexError)
    lit("\'c\'/**/", NormalErr);
    lit(" 0", NormalErr);
    lit("0 ", NormalErr);
    lit("0//", NormalErr);
    lit("3//\n4", NormalErr);
    lit("18.u8E", NormalErr);
    lit("/*a*/ //", NormalErr);
    stream("1 ) 2", NormalErr);
    stream("( x  [ ) ]", NormalErr);
    lit("1 ) 2", NormalErr);
    lit("( x  [ ) ]", NormalErr);
    // FIXME: all of the cases below should return an Err and emit no diagnostics, but don't yet.

    // emits diagnostics and returns LexError
    lit("r'r'", OtherError);
    lit("c'r'", OtherError);
    lit("\u{2000}", OtherError);

    // emits diagnostic and returns a seemingly valid tokenstream
    stream("r'r'", OtherError);
    stream("c'r'", OtherError);
    stream("\u{2000}", OtherError);

    for parse in [stream as fn(&str, Mode), lit] {
        // emits diagnostic(s), then panics
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
