// run-pass

#![allow(dead_code, non_camel_case_types)]
#![feature(macro_metavar_expr)]

macro_rules! simple_ident {
    ( $lhs:ident, $rhs:ident ) => { ${concat(lhs, rhs)} };
}

macro_rules! create_things {
    ( $lhs:ident ) => {
        struct ${concat(lhs, _separated_idents_in_a_struct)} {
            foo: i32,
            ${concat(lhs, _separated_idents_in_a_field)}: i32,
        }

        mod ${concat(lhs, _separated_idents_in_a_module)} {
            pub const FOO: () = ();
        }

        fn ${concat(lhs, _separated_idents_in_a_fn)}() {}
    };
}

create_things!(look_ma);

fn main() {
    let abcdef = 1;
    let _another = simple_ident!(abc, def);

    look_ma_separated_idents_in_a_fn();

    let _ = look_ma_separated_idents_in_a_module::FOO;

    let _ = look_ma_separated_idents_in_a_struct {
        foo: 1,
        look_ma_separated_idents_in_a_field: 2,
    };
}
