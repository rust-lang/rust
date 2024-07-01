#![feature(macro_metavar_expr_concat)]

macro_rules! wrong_concat_declarations {
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
        //~^ ERROR `${concat(..)}` currently only accepts identifiers

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
        let ${concat(1, $ident)}: () = ();
        //~^ ERROR `${concat(..)}` is not generating a valid identifier
    }};
}

macro_rules! starting_valid_unicode {
    ($ident:ident) => {{
        let ${concat('Ý', $ident)}: () = ();
    }};
}

macro_rules! starting_invalid_unicode {
    ($ident:ident) => {{
        let ${concat("\u{999999}", $ident)}: () = ();
        //~^ ERROR invalid unicode character escape
        //~| ERROR expected identifier, found
        //~| ERROR expected pattern, found
    }};
}

macro_rules! ending_number {
    ($ident:ident) => {{
        let ${concat($ident, 1)}: () = ();
    }};
}

macro_rules! ending_valid_unicode {
    ($ident:ident) => {{
        let ${concat($ident, 'Ý')}: () = ();
    }};
}

macro_rules! ending_invalid_unicode {
    ($ident:ident) => {{
        let ${concat($ident, "\u{999999}")}: () = ();
        //~^ ERROR invalid unicode character escape
        //~| ERROR expected identifier, found
        //~| ERROR expected pattern, found
    }};
}

macro_rules! empty {
    () => {{
        let ${concat("", "")}: () = ();
        //~^ ERROR `${concat(..)}` is not generating a valid identifier
    }};
}


fn main() {
    wrong_concat_declarations!(1);

    dollar_sign_without_referenced_ident!(VAR);

    starting_number!(_abc);
    starting_valid_unicode!(_abc);
    starting_invalid_unicode!(_abc);

    ending_number!(_abc);
    ending_valid_unicode!(_abc);
    ending_invalid_unicode!(_abc);

    empty!();
}
