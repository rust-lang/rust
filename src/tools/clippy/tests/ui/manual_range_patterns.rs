#![warn(clippy::manual_range_patterns)]

fn main() {
    let f = 6;

    let _ = matches!(f, 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10);
    //~^ manual_range_patterns
    let _ = matches!(f, 4 | 2 | 3 | 1 | 5 | 6 | 9 | 7 | 8 | 10);
    //~^ manual_range_patterns
    let _ = matches!(f, 4 | 2 | 3 | 1 | 5 | 6 | 9 | 8 | 10); // 7 is missing
    let _ = matches!(f, | 4);
    let _ = matches!(f, 4 | 5);
    let _ = matches!(f, 1 | 2147483647);
    let _ = matches!(f, 0 | 2147483647);
    let _ = matches!(f, -2147483647 | 2147483647);
    let _ = matches!(f, 1 | (2..=4));
    //~^ manual_range_patterns
    let _ = matches!(f, 1 | (2..4));
    //~^ manual_range_patterns
    let _ = matches!(f, (1..=10) | (2..=13) | (14..=48324728) | 48324729);
    //~^ manual_range_patterns
    let _ = matches!(f, 0 | (1..=10) | 48324730 | (2..=13) | (14..=48324728) | 48324729);
    //~^ manual_range_patterns
    let _ = matches!(f, 0..=1 | 0..=2 | 0..=3);
    //~^ manual_range_patterns
    #[allow(clippy::match_like_matches_macro)]
    let _ = match f {
        1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 => true,
        //~^ manual_range_patterns
        _ => false,
    };
    let _ = matches!(f, -1 | -5 | 3 | -2 | -4 | -3 | 0 | 1 | 2);
    //~^ manual_range_patterns
    let _ = matches!(f, -1 | -5 | 3 | -2 | -4 | -3 | 0 | 1); // 2 is missing
    let _ = matches!(f, -1_000_000..=1_000_000 | -1_000_001 | 1_000_001);
    //~^ manual_range_patterns
    let _ = matches!(f, -1_000_000..=1_000_000 | -1_000_001 | 1_000_002);

    matches!(f, 0x00 | 0x01 | 0x02 | 0x03);
    //~^ manual_range_patterns
    matches!(f, 0x00..=0x05 | 0x06 | 0x07);
    //~^ manual_range_patterns
    matches!(f, -0x09 | -0x08 | -0x07..=0x00);
    //~^ manual_range_patterns

    matches!(f, 0..5 | 5);
    //~^ manual_range_patterns
    matches!(f, 0 | 1..5);
    //~^ manual_range_patterns

    matches!(f, 0..=5 | 6..10);
    //~^ manual_range_patterns
    matches!(f, 0..5 | 5..=10);
    //~^ manual_range_patterns
    matches!(f, 5..=10 | 0..5);
    //~^ manual_range_patterns

    macro_rules! mac {
        ($e:expr) => {
            matches!($e, 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10)
            //~^ manual_range_patterns
        };
    }
    mac!(f);

    #[rustfmt::skip]
    let _ = match f {
        | 2..=15 => 4,
        | 241..=254 => 5,
        | 255 => 6,
        | _ => 7,
    };

    char_literal();
    byte_literal();
}

fn char_literal() {
    let ch = '0';

    matches!(ch, 'a' | 'b' | 'c' | 'd' | 'e');
    //~^ manual_range_patterns
    matches!(ch, '0' | '1' | '2' | '3' | '4');
    //~^ manual_range_patterns

    // FIXME: Reduce to 'A'..='Z' | '[' | '\\' | ']' | '^' | '_' | '`' | 'a'..='z'
    #[rustfmt::skip]
    matches!(ch, 'A' | 'B' | 'C' | 'D' | 'E' | 'F' | 'G' | 'H' | 'I' | 'J' | 'K' | 'L' | 'M' | 'N' |
        'O' | 'P' | 'Q' | 'R' | 'S' | 'T' | 'U' | 'V' | 'W' | 'X' | 'Y' | 'Z' | '[' | '\\' | ']' |
        '^' | '_' | '`' | 'a' | 'b' | 'c' | 'd' | 'e' | 'f' | 'g' | 'h' | 'i' | 'j' | 'k' | 'l' |
        'm' | 'n' | 'o' | 'p' | 'q' | 'r' | 's' | 't' | 'u' | 'v' | 'w' | 'x' | 'y' | 'z');

    matches!(ch, '0' | '1' | '2' | '4'); // '3' is missing
    matches!(ch, 'A' | 'B' | 'c' | 'd' | 'e'); // 'C'..'c' is missing
    matches!(ch, '<' | '=' | '>'); // range is not intuitive
    matches!(ch, '/' | '0'..='9'); // range is not intuitive
}
fn byte_literal() {
    let byte = b'0';

    matches!(byte, b'a' | b'b' | b'c' | b'd' | b'e');
    //~^ manual_range_patterns
    matches!(byte, b'0' | b'1' | b'2' | b'3' | b'4');
    //~^ manual_range_patterns

    matches!(byte, b'0' | b'1' | b'2' | b'4'); // b'3' is missing
    matches!(byte, b'A' | b'B' | b'c' | b'd' | b'e'); // b'C'..b'c' is missing
    matches!(byte, b'<' | b'=' | b'>'); // range is not intuitive
    matches!(byte, b'/' | b'0'..=b'9'); // range is not intuitive
}
