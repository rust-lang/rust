//@ run-pass

#![feature(macro_metavar_expr_concat)]

macro_rules! turn_to_page {
    ($ident:ident) => {
        const ${concat("Ḧ", $ident)}: i32 = 394;
    };
}

fn main() {
    turn_to_page!(P);
    assert_eq!(ḦP, 394);
}
