//@ run-pass

#![feature(macro_metavar_expr_concat)]

macro_rules! turn_to_page {
    ($ident:ident, $literal:literal, $tt:tt) => {
        const ${concat("Ḧ", $ident)}: i32 = 394;
        const ${concat("Ḧ", $literal)}: i32 = 394;
        const ${concat("Ḧ", $tt)}: i32 = 394;
    };
}

fn main() {
    turn_to_page!(P1, "Ṕ2", Ṕ);
    assert_eq!(ḦṔ, 394);
    assert_eq!(ḦP1, 394);
    assert_eq!(ḦṔ2, 394);

}
