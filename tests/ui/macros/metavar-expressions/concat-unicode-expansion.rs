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
    //~^ WARN: identifier contains an uncommon character: 'Ṕ'
    assert_eq!(ḦṔ, 394);
    //~^ WARN: identifier contains uncommon characters: 'Ḧ' and 'Ṕ'
    assert_eq!(ḦP1, 394);
    //~^ WARN: identifier contains an uncommon character: 'Ḧ'
    assert_eq!(ḦṔ2, 394);
    //~^ WARN: identifier contains uncommon characters: 'Ḧ' and 'Ṕ'
}
