// run-rustfix
#![warn(clippy::unnecessary_trailing_comma)]

fn main() {}

// fmt breaks - https://github.com/rust-lang/rustfmt/issues/6797
#[rustfmt::skip]
fn simple() {
    println!["Foo(,)"];
    println!("Foo" , ); //~ unnecessary_trailing_comma
    println!{"Foo" , }; //~ unnecessary_trailing_comma
    println!["Foo" , ]; //~ unnecessary_trailing_comma
    println!("Foo={}",   1  ,  ); //~ unnecessary_trailing_comma
    println!(concat!("b", "o", "o")  , ); //~ unnecessary_trailing_comma
    println!("Foo(,)",); //~ unnecessary_trailing_comma
    println!("Foo[,]" , ); //~ unnecessary_trailing_comma
    println!["Foo(,)", ]; //~ unnecessary_trailing_comma
    println!["Foo[,]", ]; //~ unnecessary_trailing_comma
    println!["Foo{{,}}", ]; //~ unnecessary_trailing_comma
    println!{"Foo{{,}}", }; //~ unnecessary_trailing_comma
    println!{"Foo(,)", }; //~ unnecessary_trailing_comma
    println!{"Foo[,]", }; //~ unnecessary_trailing_comma
    println!["Foo(,", ]; //~ unnecessary_trailing_comma
    println!["Foo[,", ]; //~ unnecessary_trailing_comma
    println!["Foo{{,}}", ]; //~ unnecessary_trailing_comma
    println!{"Foo{{,}}", }; //~ unnecessary_trailing_comma
    println!{"Foo(,", }; //~ unnecessary_trailing_comma
    println!{"Foo[,", }; //~ unnecessary_trailing_comma

    // This should eventually work, but requires more work
    println!(concat!("Foo", "=", "{}"), 1,);
    println!("No params", /*"a,){ */);
    println!("No params" /* "a,){*/, /*"a,){ */);

    // No trailing comma - no lint
    println!("{}", 1);
    println!(concat!("b", "o", "o"));
    println!(concat!("Foo", "=", "{}"), 1);

    println!("Foo" );
    println!{"Foo" };
    println!["Foo" ];
    println!("Foo={}", 1);
    println!(concat!("b", "o", "o"));
    println!("Foo(,)");
    println!("Foo[,]");
    println!["Foo(,)"];
    println!["Foo[,]"];
    println!["Foo{{,}}"];
    println!{"Foo{{,}}"};
    println!{"Foo(,)"};
    println!{"Foo[,]"};
    println!["Foo(,"];
    println!["Foo[,"];
    println!["Foo{{,}}"];
    println!{"Foo{{,}}"};
    println!{"Foo(,"};
    println!{"Foo[,"};

    // Multi-line macro - must NOT lint (single-line only)
    println!(
        "very long string to prevent fmt from making it into a single line: {}",
        1,
    );

    print!("{}"
        , 1
        ,);
}

// The macro invocation itself should never be fixed
// The call to println! on the other hand might be ok to suggest in the future

macro_rules! from_macro {
    (0,) => {
        println!("Foo",);
    };
    (1,) => {
        println!("Foo={}", 1,);
    };
}

fn from_macro() {
    from_macro!(0,);
    from_macro!(1,);
}
