//@ run-pass

#![feature(macro_metavar_expr_concat)]

macro_rules! escape_literal {
    ($literal:literal, $concatted:literal) => {
        let lit = ${concat_str("_", "\"foo", $literal)};
        assert_eq!(lit, concat!("_", "\"foo", $literal));
        assert_eq!(lit, $concatted);

    }
}

fn main(){
    escape_literal!("\u{00BD}", "_\"foo\u{00BD}");
    escape_literal!("\x41", "_\"foo\x41");
    escape_literal!("🤷", "_\"foo🤷");
    escape_literal!("\u{1F980}", "_\"foo🦀");
    escape_literal!("aaa \"bbb\" ccc", "_\"fooaaa \"bbb\" ccc");
}
