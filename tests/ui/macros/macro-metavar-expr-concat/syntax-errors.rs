#![feature(macro_metavar_expr_concat)]

macro_rules! wrong_concat_declarations {
    ($ex:expr) => {
        ${concat()}
        //~^ ERROR expected identifier

        ${concat(aaaa)}
        //~^ ERROR `concat` must have at least two elements

        ${concat(aaaa,)}
        //~^ ERROR expected identifier

        ${concat(aaaa, 1)}
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

        ${concat($ex, aaaa, 123)}
        //~^ ERROR expected identifier
    };
}

macro_rules! dollar_sign_without_referenced_ident {
    ($ident:ident) => {
        const ${concat(FOO, $foo)}: i32 = 2;
        //~^ ERROR variable `foo` is not recognized in meta-variable expression
    };
}

fn main() {
    wrong_concat_declarations!(1);

    dollar_sign_without_referenced_ident!(VAR);
}
