//@ edition: 2021

#![feature(macro_metavar_expr_concat)]

macro_rules! syntax_errors {
    ($ex:expr) => {
        ${concat()}
        //~^ ERROR expected identifier

        ${concat(aaaa)}
        //~^ ERROR `concat` must have at least two elements

        ${concat(aaaa,)}
        //~^ ERROR expected identifier

        ${concat(_, aaaa)}

        ${concat(aaaa aaaa)}
        //~^ ERROR expected comma

        ${concat($ex)}
        //~^ ERROR `concat` must have at least two elements

        ${concat($ex, aaaa)}
        //~^ ERROR metavariables of `${concat(..)}` must be of type

        ${concat($ex, aaaa 123)}
        //~^ ERROR expected comma

        ${concat($ex, aaaa,)}
        //~^ ERROR expected identifier
    };
}

macro_rules! dollar_sign_without_referenced_ident {
    ($ident:ident) => {
        const ${concat(FOO, $foo)}: i32 = 2;
        //~^ ERROR variable `foo` is not recognized in meta-variable expression
    };
}

macro_rules! starting_number {
    ($ident:ident) => {{
        let ${concat("1", $ident)}: () = ();
        //~^ ERROR `${concat(..)}` is not generating a valid identifier
    }};
}

macro_rules! starting_valid_unicode {
    ($ident:ident) => {{
        let ${concat("Ý", $ident)}: () = ();
    }};
}

macro_rules! starting_invalid_unicode {
    ($ident:ident) => {{
        let ${concat("\u{00BD}", $ident)}: () = ();
        //~^ ERROR `${concat(..)}` is not generating a valid identifier
    }};
}

macro_rules! ending_number {
    ($ident:ident) => {{
        let ${concat($ident, "1")}: () = ();
    }};
}

macro_rules! ending_valid_unicode {
    ($ident:ident) => {{
        let ${concat($ident, "Ý")}: () = ();
    }};
}

macro_rules! ending_invalid_unicode {
    ($ident:ident) => {{
        let ${concat($ident, "\u{00BD}")}: () = ();
        //~^ ERROR `${concat(..)}` is not generating a valid identifier
    }};
}

macro_rules! empty {
    () => {{
        let ${concat("", "")}: () = ();
        //~^ ERROR `${concat(..)}` is not generating a valid identifier
    }};
}

macro_rules! unsupported_literals {
    ($ident:ident) => {{
        let ${concat(_a, 'b')}: () = ();
        //~^ ERROR expected identifier or string literal
        //~| ERROR expected pattern
        let ${concat(_a, 1)}: () = ();
        //~^ ERROR expected identifier or string literal
        let ${concat(_a, 1.5)}: () = ();
        //~^ ERROR expected identifier or string literal
        let ${concat(_a, c"hi")}: () = ();
        //~^ ERROR expected identifier or string literal
        let ${concat(_a, b"hi")}: () = ();
        //~^ ERROR expected identifier or string literal
        let ${concat(_a, b'b')}: () = ();
        //~^ ERROR expected identifier or string literal
        let ${concat(_a, b'b')}: () = ();
        //~^ ERROR expected identifier or string literal

        let ${concat($ident, 'b')}: () = ();
        //~^ ERROR expected identifier or string literal
        let ${concat($ident, 1)}: () = ();
        //~^ ERROR expected identifier or string literal
        let ${concat($ident, 1.5)}: () = ();
        //~^ ERROR expected identifier or string literal
        let ${concat($ident, c"hi")}: () = ();
        //~^ ERROR expected identifier or string literal
        let ${concat($ident, b"hi")}: () = ();
        //~^ ERROR expected identifier or string literal
        let ${concat($ident, b'b')}: () = ();
        //~^ ERROR expected identifier or string literal
        let ${concat($ident, b'b')}: () = ();
        //~^ ERROR expected identifier or string literal
    }};
}

macro_rules! bad_literal_string {
    ($literal:literal) => {
        const ${concat(_foo, $literal)}: () = ();
        //~^ ERROR `${concat(..)}` is not generating a valid identifier
        //~| ERROR `${concat(..)}` is not generating a valid identifier
        //~| ERROR `${concat(..)}` is not generating a valid identifier
        //~| ERROR `${concat(..)}` is not generating a valid identifier
        //~| ERROR `${concat(..)}` is not generating a valid identifier
        //~| ERROR `${concat(..)}` is not generating a valid identifier
        //~| ERROR `${concat(..)}` is not generating a valid identifier
    }
}

macro_rules! bad_literal_non_string {
    ($literal:literal) => {
        const ${concat(_foo, $literal)}: () = ();
        //~^ ERROR metavariables of `${concat(..)}` must be of type
        //~| ERROR metavariables of `${concat(..)}` must be of type
        //~| ERROR metavariables of `${concat(..)}` must be of type
        //~| ERROR metavariables of `${concat(..)}` must be of type
        //~| ERROR metavariables of `${concat(..)}` must be of type
    }
}

macro_rules! bad_tt_literal {
    ($tt:tt) => {
        const ${concat(_foo, $tt)}: () = ();
        //~^ ERROR metavariables of `${concat(..)}` must be of type
        //~| ERROR metavariables of `${concat(..)}` must be of type
        //~| ERROR metavariables of `${concat(..)}` must be of type
    }
}

fn main() {
    syntax_errors!(1);

    dollar_sign_without_referenced_ident!(VAR);

    starting_number!(_abc);
    starting_valid_unicode!(_abc);
    starting_invalid_unicode!(_abc);

    ending_number!(_abc);
    ending_valid_unicode!(_abc);
    ending_invalid_unicode!(_abc);
    unsupported_literals!(_abc);

    empty!();

    bad_literal_string!("\u{00BD}");
    bad_literal_string!("\x41");
    bad_literal_string!("🤷");
    bad_literal_string!("d[-_-]b");

    bad_literal_string!("-1");
    bad_literal_string!("1.0");
    bad_literal_string!("'1'");

    bad_literal_non_string!(1);
    bad_literal_non_string!(-1);
    bad_literal_non_string!(1.0);
    bad_literal_non_string!('1');
    bad_literal_non_string!(false);

    bad_tt_literal!(1);
    bad_tt_literal!(1.0);
    bad_tt_literal!('1');
}
