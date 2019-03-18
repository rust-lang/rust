// See comments in `u8::to_ascii_uppercase` in `src/libcore/num/mod.rs`.
fn branchless_to_ascii_upper_case(byte: u8) -> u8 {
    byte &
    !(
        (
            byte.wrapping_add(0x1f) &
            !byte.wrapping_add(0x05) &
            0x80
        ) >> 2
    )
}


macro_rules! benches {
    ($( fn $name: ident($arg: ident: &mut [u8]) $body: block )+) => {
        benches!(mod short SHORT $($name $arg $body)+);
        benches!(mod medium MEDIUM $($name $arg $body)+);
        benches!(mod long LONG $($name $arg $body)+);
    };

    (mod $mod_name: ident $input: ident $($name: ident $arg: ident $body: block)+) => {
        mod $mod_name {
            use super::*;

            $(
                #[bench]
                fn $name(bencher: &mut Bencher) {
                    bencher.bytes = $input.len() as u64;
                    bencher.iter(|| {
                        let mut vec = $input.as_bytes().to_vec();
                        {
                            let $arg = &mut vec[..];
                            $body
                        }
                        vec
                    })
                }
            )+
        }
    }
}

use test::black_box;
use test::Bencher;

benches! {
    fn bench00_alloc_only(_bytes: &mut [u8]) {}

    fn bench01_black_box_read_each_byte(bytes: &mut [u8]) {
        for byte in bytes {
            black_box(*byte);
        }
    }

    fn bench02_lookup(bytes: &mut [u8]) {
        for byte in bytes {
            *byte = ASCII_UPPERCASE_MAP[*byte as usize]
        }
    }

    fn bench03_branch_and_subtract(bytes: &mut [u8]) {
        for byte in bytes {
            *byte = if b'a' <= *byte && *byte <= b'z' {
                *byte - b'a' + b'A'
            } else {
                *byte
            }
        }
    }

    fn bench04_branch_and_mask(bytes: &mut [u8]) {
        for byte in bytes {
            *byte = if b'a' <= *byte && *byte <= b'z' {
                *byte & !0x20
            } else {
                *byte
            }
        }
    }

    fn bench05_branchless(bytes: &mut [u8]) {
        for byte in bytes {
            *byte = branchless_to_ascii_upper_case(*byte)
        }
    }

    fn bench06_libcore(bytes: &mut [u8]) {
        bytes.make_ascii_uppercase()
    }

    fn bench07_fake_simd_u32(bytes: &mut [u8]) {
        let (before, aligned, after) = unsafe {
            bytes.align_to_mut::<u32>()
        };
        for byte in before {
            *byte = branchless_to_ascii_upper_case(*byte)
        }
        for word in aligned {
            // FIXME: this is incorrect for some byte values:
            // addition within a byte can carry/overflow into the next byte.
            // Test case: b"\xFFz  "
            *word &= !(
                (
                    word.wrapping_add(0x1f1f1f1f) &
                    !word.wrapping_add(0x05050505) &
                    0x80808080
                ) >> 2
            )
        }
        for byte in after {
            *byte = branchless_to_ascii_upper_case(*byte)
        }
    }

    fn bench08_fake_simd_u64(bytes: &mut [u8]) {
        let (before, aligned, after) = unsafe {
            bytes.align_to_mut::<u64>()
        };
        for byte in before {
            *byte = branchless_to_ascii_upper_case(*byte)
        }
        for word in aligned {
            // FIXME: like above, this is incorrect for some byte values.
            *word &= !(
                (
                    word.wrapping_add(0x1f1f1f1f_1f1f1f1f) &
                    !word.wrapping_add(0x05050505_05050505) &
                    0x80808080_80808080
                ) >> 2
            )
        }
        for byte in after {
            *byte = branchless_to_ascii_upper_case(*byte)
        }
    }
}

macro_rules! repeat {
    ($s: expr) => { concat!($s, $s, $s, $s, $s, $s, $s, $s, $s, $s) }
}

const SHORT: &'static str = "Alice's";
const MEDIUM: &'static str = "Alice's Adventures in Wonderland";
const LONG: &'static str = repeat!(r#"
    La Guida di Bragia, a Ballad Opera for the Marionette Theatre (around 1850)
    Alice's Adventures in Wonderland (1865)
    Phantasmagoria and Other Poems (1869)
    Through the Looking-Glass, and What Alice Found There
        (includes "Jabberwocky" and "The Walrus and the Carpenter") (1871)
    The Hunting of the Snark (1876)
    Rhyme? And Reason? (1883) – shares some contents with the 1869 collection,
        including the long poem "Phantasmagoria"
    A Tangled Tale (1885)
    Sylvie and Bruno (1889)
    Sylvie and Bruno Concluded (1893)
    Pillow Problems (1893)
    What the Tortoise Said to Achilles (1895)
    Three Sunsets and Other Poems (1898)
    The Manlet (1903)[106]
"#);

const ASCII_UPPERCASE_MAP: [u8; 256] = [
    0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07,
    0x08, 0x09, 0x0a, 0x0b, 0x0c, 0x0d, 0x0e, 0x0f,
    0x10, 0x11, 0x12, 0x13, 0x14, 0x15, 0x16, 0x17,
    0x18, 0x19, 0x1a, 0x1b, 0x1c, 0x1d, 0x1e, 0x1f,
    b' ', b'!', b'"', b'#', b'$', b'%', b'&', b'\'',
    b'(', b')', b'*', b'+', b',', b'-', b'.', b'/',
    b'0', b'1', b'2', b'3', b'4', b'5', b'6', b'7',
    b'8', b'9', b':', b';', b'<', b'=', b'>', b'?',
    b'@', b'A', b'B', b'C', b'D', b'E', b'F', b'G',
    b'H', b'I', b'J', b'K', b'L', b'M', b'N', b'O',
    b'P', b'Q', b'R', b'S', b'T', b'U', b'V', b'W',
    b'X', b'Y', b'Z', b'[', b'\\', b']', b'^', b'_',
    b'`',

          b'A', b'B', b'C', b'D', b'E', b'F', b'G',
    b'H', b'I', b'J', b'K', b'L', b'M', b'N', b'O',
    b'P', b'Q', b'R', b'S', b'T', b'U', b'V', b'W',
    b'X', b'Y', b'Z',

                      b'{', b'|', b'}', b'~', 0x7f,
    0x80, 0x81, 0x82, 0x83, 0x84, 0x85, 0x86, 0x87,
    0x88, 0x89, 0x8a, 0x8b, 0x8c, 0x8d, 0x8e, 0x8f,
    0x90, 0x91, 0x92, 0x93, 0x94, 0x95, 0x96, 0x97,
    0x98, 0x99, 0x9a, 0x9b, 0x9c, 0x9d, 0x9e, 0x9f,
    0xa0, 0xa1, 0xa2, 0xa3, 0xa4, 0xa5, 0xa6, 0xa7,
    0xa8, 0xa9, 0xaa, 0xab, 0xac, 0xad, 0xae, 0xaf,
    0xb0, 0xb1, 0xb2, 0xb3, 0xb4, 0xb5, 0xb6, 0xb7,
    0xb8, 0xb9, 0xba, 0xbb, 0xbc, 0xbd, 0xbe, 0xbf,
    0xc0, 0xc1, 0xc2, 0xc3, 0xc4, 0xc5, 0xc6, 0xc7,
    0xc8, 0xc9, 0xca, 0xcb, 0xcc, 0xcd, 0xce, 0xcf,
    0xd0, 0xd1, 0xd2, 0xd3, 0xd4, 0xd5, 0xd6, 0xd7,
    0xd8, 0xd9, 0xda, 0xdb, 0xdc, 0xdd, 0xde, 0xdf,
    0xe0, 0xe1, 0xe2, 0xe3, 0xe4, 0xe5, 0xe6, 0xe7,
    0xe8, 0xe9, 0xea, 0xeb, 0xec, 0xed, 0xee, 0xef,
    0xf0, 0xf1, 0xf2, 0xf3, 0xf4, 0xf5, 0xf6, 0xf7,
    0xf8, 0xf9, 0xfa, 0xfb, 0xfc, 0xfd, 0xfe, 0xff,
];

