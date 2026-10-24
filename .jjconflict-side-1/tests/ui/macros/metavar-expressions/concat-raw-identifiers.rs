#![feature(macro_metavar_expr_concat)]

macro_rules! idents_01 {
    ($rhs:ident) => {
        let ${concat(abc, $rhs)}: () = ();
        //~^ ERROR `${concat(..)}` currently does not support raw identifiers
    };
}

macro_rules! idents_10 {
    ($lhs:ident) => {
        let ${concat($lhs, abc)}: () = ();
        //~^ ERROR `${concat(..)}` currently does not support raw identifiers
    };
}

macro_rules! idents_11 {
    ($lhs:ident, $rhs:ident) => {
        let ${concat($lhs, $rhs)}: () = ();
        //~^ ERROR `${concat(..)}` currently does not support raw identifiers
        //~| ERROR `${concat(..)}` currently does not support raw identifiers
        //~| ERROR `${concat(..)}` currently does not support raw identifiers
    };
}

macro_rules! no_params {
    () => {
        let ${concat(r#abc, abc)}: () = ();
        //~^ ERROR expected identifier or string literal
        //~| ERROR expected pattern, found `$`

        let ${concat(abc, r#abc)}: () = ();
        //~^ ERROR expected identifier or string literal

        let ${concat(r#abc, r#abc)}: () = ();
        //~^ ERROR expected identifier or string literal
    };
}

macro_rules! tts_01 {
    ($rhs:tt) => {
        let ${concat(abc, $rhs)}: () = ();
        //~^ ERROR `${concat(..)}` currently does not support raw identifiers
    };
}

macro_rules! tts_10 {
    ($lhs:tt) => {
        let ${concat($lhs, abc)}: () = ();
        //~^ ERROR `${concat(..)}` currently does not support raw identifiers
    };
}

macro_rules! tts_11 {
    ($lhs:tt, $rhs:tt) => {
        let ${concat($lhs, $rhs)}: () = ();
        //~^ ERROR `${concat(..)}` currently does not support raw identifiers
        //~| ERROR `${concat(..)}` currently does not support raw identifiers
        //~| ERROR `${concat(..)}` currently does not support raw identifiers
    };
}

fn main() {
    idents_01!(r#_c);

    idents_10!(r#_c);

    idents_11!(r#_c, d);
    idents_11!(_e, r#f);
    idents_11!(r#_g, r#h);

    tts_01!(r#_c);

    tts_10!(r#_c);

    tts_11!(r#_c, d);
    tts_11!(_e, r#f);
    tts_11!(r#_g, r#h);

    no_params!();
}
