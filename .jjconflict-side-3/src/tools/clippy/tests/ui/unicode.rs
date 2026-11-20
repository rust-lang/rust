fn zero() {
    #![warn(clippy::invisible_characters)]

    print!("Here >​< is a ZWS, and ​another");
    //~^ invisible_characters
    print!("This\u{200B}is\u{200B}fine");
    print!("Here >­< is a SHY, and ­another");
    //~^ invisible_characters
    print!("This\u{ad}is\u{ad}fine");
    print!("Here >⁠< is a WJ, and ⁠another");
    //~^ invisible_characters
    print!("This\u{2060}is\u{2060}fine");
}

fn canon() {
    #![warn(clippy::unicode_not_nfc)]

    print!("̀àh?");
    //~^ unicode_not_nfc
    print!("a\u{0300}h?"); // also ok
}

mod non_ascii_literal {
    #![warn(clippy::non_ascii_literal)]

    fn uni() {
        print!("Üben!");
        //~^ non_ascii_literal
        print!("\u{DC}ben!"); // this is ok
    }

    // issue 8013
    fn single_quote() {
        const _EMPTY_BLOCK: char = '▱';
        //~^ non_ascii_literal
        const _FULL_BLOCK: char = '▰';
        //~^ non_ascii_literal
    }

    #[test]
    pub fn issue_7739() {
        // Ryū crate: https://github.com/dtolnay/ryu
    }

    mod issue_8263 {
        // Re-allow for a single test
        #[test]
        #[allow(clippy::non_ascii_literal)]
        fn allowed() {
            let _ = "悲しいかな、ここに日本語を書くことはできない。";
        }

        #[test]
        fn denied() {
            let _ = "悲しいかな、ここに日本語を書くことはできない。";
            //~^ non_ascii_literal
        }
    }
}

mod ascii_macro_with_non_ascii_value {
    // The source is pure ASCII, but the value contains non-ASCII, invisible, and
    // non-NFC characters. The lints check the source snippet, not the value, so
    // none of them may fire here.
    #![deny(clippy::invisible_characters, clippy::non_ascii_literal, clippy::unicode_not_nfc)]

    macro_rules! non_ascii_value {
        () => {
            "\u{00E9}\u{200B}a\u{0300}"
        };
    }

    fn no_lint() {
        let _ = non_ascii_value!();
    }
}

mod ascii_value_from_non_ascii_snippet {
    // The reverse: `file!()` produces an ASCII value (the file path), but the
    // literal's span covers the whole macro invocation, so the snippet is
    // non-ASCII and the lint must fire. Checking the value would miss it.
    #![warn(clippy::non_ascii_literal)]

    macro_rules! with_location {
        ($($arg:tt)*) => {
            let _ = file!();
        };
    }

    fn lint() {
        with_location!("héllo");
        //~^ non_ascii_literal
    }
}

fn main() {}
