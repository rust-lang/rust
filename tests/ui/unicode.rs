//@run-rustfix
#![allow(dead_code)]

#[warn(clippy::invisible_characters)]
fn zero() {
    print!("Here >​< is a ZWS, and ​another");
    print!("This\u{200B}is\u{200B}fine");
    print!("Here >­< is a SHY, and ­another");
    print!("This\u{ad}is\u{ad}fine");
    print!("Here >⁠< is a WJ, and ⁠another");
    print!("This\u{2060}is\u{2060}fine");
}

#[warn(clippy::unicode_not_nfc)]
fn canon() {
    print!("̀àh?");
    print!("a\u{0300}h?"); // also ok
}

mod non_ascii_literal {
    #![deny(clippy::non_ascii_literal)]

    fn uni() {
        print!("Üben!");
        print!("\u{DC}ben!"); // this is ok
    }

    // issue 8013
    fn single_quote() {
        const _EMPTY_BLOCK: char = '▱';
        const _FULL_BLOCK: char = '▰';
    }

    #[test]
    pub fn issue_7739() {
        // Ryū crate: https://github.com/dtolnay/ryu
    }

    mod issue_8263 {
        #![deny(clippy::non_ascii_literal)]

        // Re-allow for a single test
        #[test]
        #[allow(clippy::non_ascii_literal)]
        fn allowed() {
            let _ = "悲しいかな、ここに日本語を書くことはできない。";
        }

        #[test]
        fn denied() {
            let _ = "悲しいかな、ここに日本語を書くことはできない。";
        }
    }
}

fn main() {
    zero();
    canon();
}
