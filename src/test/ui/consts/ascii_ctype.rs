// run-pass

macro_rules! suite {
    ( $( $fn:ident => [$a:ident, $A:ident, $nine:ident, $dot:ident, $space:ident]; )* ) => {
        $(
            mod $fn {
                const CHAR_A_LOWER: bool = 'a'.$fn();
                const CHAR_A_UPPER: bool = 'A'.$fn();
                const CHAR_NINE: bool = '9'.$fn();
                const CHAR_DOT: bool = '.'.$fn();
                const CHAR_SPACE: bool = ' '.$fn();

                const U8_A_LOWER: bool = b'a'.$fn();
                const U8_A_UPPER: bool = b'A'.$fn();
                const U8_NINE: bool = b'9'.$fn();
                const U8_DOT: bool = b'.'.$fn();
                const U8_SPACE: bool = b' '.$fn();

                pub fn run() {
                    assert_eq!(CHAR_A_LOWER, $a);
                    assert_eq!(CHAR_A_UPPER, $A);
                    assert_eq!(CHAR_NINE, $nine);
                    assert_eq!(CHAR_DOT, $dot);
                    assert_eq!(CHAR_SPACE, $space);

                    assert_eq!(U8_A_LOWER, $a);
                    assert_eq!(U8_A_UPPER, $A);
                    assert_eq!(U8_NINE, $nine);
                    assert_eq!(U8_DOT, $dot);
                    assert_eq!(U8_SPACE, $space);
                }
            }
        )*

        fn main() {
            $( $fn::run(); )*
        }
    }
}

suite! {
    //                        'a'    'A'    '9'    '.'    ' '
    is_ascii_alphabetic   => [true,  true,  false, false, false];
    is_ascii_uppercase    => [false, true,  false, false, false];
    is_ascii_lowercase    => [true,  false, false, false, false];
    is_ascii_alphanumeric => [true,  true,  true,  false, false];
    is_ascii_digit        => [false, false, true,  false, false];
    is_ascii_hexdigit     => [true,  true,  true,  false, false];
    is_ascii_punctuation  => [false, false, false, true,  false];
    is_ascii_graphic      => [true,  true,  true,  true,  false];
    is_ascii_whitespace   => [false, false, false, false, true];
    is_ascii_control      => [false, false, false, false, false];
}
