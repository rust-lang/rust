mod is_ascii;

// Lower-case ASCII 'a' is the first byte that has its highest bit set
// after wrap-adding 0x1F:
//
//     b'a' + 0x1F == 0x80 == 0b1000_0000
//     b'z' + 0x1F == 0x98 == 0b1001_1000
//
// Lower-case ASCII 'z' is the last byte that has its highest bit unset
// after wrap-adding 0x05:
//
//     b'a' + 0x05 == 0x66 == 0b0110_0110
//     b'z' + 0x05 == 0x7F == 0b0111_1111
//
// … except for 0xFB to 0xFF, but those are in the range of bytes
// that have the highest bit unset again after adding 0x1F.
//
// So `(byte + 0x1f) & !(byte + 5)` has its highest bit set
// iff `byte` is a lower-case ASCII letter.
//
// Lower-case ASCII letters all have the 0x20 bit set.
// (Two positions right of 0x80, the highest bit.)
// Unsetting that bit produces the same letter, in upper-case.
//
// Therefore:
fn branchless_to_ascii_upper_case(byte: u8) -> u8 {
    byte & !((byte.wrapping_add(0x1f) & !byte.wrapping_add(0x05) & 0x80) >> 2)
}

macro_rules! benches {
    ($( fn $name: ident($arg: ident: &mut [u8]) $body: block )+ @iter $( $is_: ident, )+) => {
        benches! {@
            $( fn $name($arg: &mut [u8]) $body )+
            $( fn $is_(bytes: &mut [u8]) { bytes.iter().all(u8::$is_) } )+
        }
    };

    (@$( fn $name: ident($arg: ident: &mut [u8]) $body: block )+) => {
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
                            black_box($body);
                        }
                        vec
                    })
                }
            )+
        }
    }
}

use std::fmt::Write;

use test::{Bencher, black_box};

const ASCII_CASE_MASK: u8 = 0b0010_0000;

benches! {
    fn case00_alloc_only(_bytes: &mut [u8]) {}

    fn case01_black_box_read_each_byte(bytes: &mut [u8]) {
        for byte in bytes {
            black_box(*byte);
        }
    }

    fn case02_lookup_table(bytes: &mut [u8]) {
        for byte in bytes {
            *byte = ASCII_UPPERCASE_MAP[*byte as usize]
        }
    }

    fn case03_branch_and_subtract(bytes: &mut [u8]) {
        for byte in bytes {
            *byte = if b'a' <= *byte && *byte <= b'z' {
                *byte - b'a' + b'A'
            } else {
                *byte
            }
        }
    }

    fn case04_branch_and_mask(bytes: &mut [u8]) {
        for byte in bytes {
            *byte = if b'a' <= *byte && *byte <= b'z' {
                *byte & !0x20
            } else {
                *byte
            }
        }
    }

    fn case05_branchless(bytes: &mut [u8]) {
        for byte in bytes {
            *byte = branchless_to_ascii_upper_case(*byte)
        }
    }

    fn case06_libcore(bytes: &mut [u8]) {
        bytes.make_ascii_uppercase()
    }

    fn case07_fake_simd_u32(bytes: &mut [u8]) {
        // SAFETY: transmuting a sequence of `u8` to `u32` is always fine
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

    fn case08_fake_simd_u64(bytes: &mut [u8]) {
        // SAFETY: transmuting a sequence of `u8` to `u64` is always fine
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

    fn case09_mask_mult_bool_branchy_lookup_table(bytes: &mut [u8]) {
        fn is_ascii_lowercase(b: u8) -> bool {
            if b >= 0x80 { return false }
            match ASCII_CHARACTER_CLASS[b as usize] {
                L | Lx => true,
                _ => false,
            }
        }
        for byte in bytes {
            *byte &= !(0x20 * (is_ascii_lowercase(*byte) as u8))
        }
    }

    fn case10_mask_mult_bool_lookup_table(bytes: &mut [u8]) {
        fn is_ascii_lowercase(b: u8) -> bool {
            match ASCII_CHARACTER_CLASS[b as usize] {
                L | Lx => true,
                _ => false
            }
        }
        for byte in bytes {
            *byte &= !(0x20 * (is_ascii_lowercase(*byte) as u8))
        }
    }

    fn case11_mask_mult_bool_match_range(bytes: &mut [u8]) {
        fn is_ascii_lowercase(b: u8) -> bool {
            match b {
                b'a'..=b'z' => true,
                _ => false
            }
        }
        for byte in bytes {
            *byte &= !(0x20 * (is_ascii_lowercase(*byte) as u8))
        }
    }

    fn case12_mask_shifted_bool_match_range(bytes: &mut [u8]) {
        fn is_ascii_lowercase(b: u8) -> bool {
            match b {
                b'a'..=b'z' => true,
                _ => false
            }
        }
        for byte in bytes {
            *byte &= !((is_ascii_lowercase(*byte) as u8) * ASCII_CASE_MASK)
        }
    }

    fn case13_subtract_shifted_bool_match_range(bytes: &mut [u8]) {
        fn is_ascii_lowercase(b: u8) -> bool {
            match b {
                b'a'..=b'z' => true,
                _ => false
            }
        }
        for byte in bytes {
            *byte -= (is_ascii_lowercase(*byte) as u8) * ASCII_CASE_MASK
        }
    }

    fn case14_subtract_multiplied_bool_match_range(bytes: &mut [u8]) {
        fn is_ascii_lowercase(b: u8) -> bool {
            match b {
                b'a'..=b'z' => true,
                _ => false
            }
        }
        for byte in bytes {
            *byte -= (b'a' - b'A') * is_ascii_lowercase(*byte) as u8
        }
    }

    @iter

    is_ascii,
    is_ascii_alphabetic,
    is_ascii_uppercase,
    is_ascii_lowercase,
    is_ascii_alphanumeric,
    is_ascii_digit,
    is_ascii_hexdigit,
    is_ascii_punctuation,
    is_ascii_graphic,
    is_ascii_whitespace,
    is_ascii_control,
}

macro_rules! repeat {
    ($s: expr) => {
        concat!($s, $s, $s, $s, $s, $s, $s, $s, $s, $s)
    };
}

const SHORT: &str = "Alice's";
const MEDIUM: &str = "Alice's Adventures in Wonderland";
const LONG: &str = repeat!(
    r#"
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
"#
);

#[rustfmt::skip]
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

enum AsciiCharacterClass {
    C,  // control
    Cw, // control whitespace
    W,  // whitespace
    D,  // digit
    L,  // lowercase
    Lx, // lowercase hex digit
    U,  // uppercase
    Ux, // uppercase hex digit
    P,  // punctuation
    N,  // Non-ASCII
}
use self::AsciiCharacterClass::*;

#[rustfmt::skip]
static ASCII_CHARACTER_CLASS: [AsciiCharacterClass; 256] = [
//  _0 _1 _2 _3 _4 _5 _6 _7 _8 _9 _a _b _c _d _e _f
    C, C, C, C, C, C, C, C, C, Cw,Cw,C, Cw,Cw,C, C, // 0_
    C, C, C, C, C, C, C, C, C, C, C, C, C, C, C, C, // 1_
    W, P, P, P, P, P, P, P, P, P, P, P, P, P, P, P, // 2_
    D, D, D, D, D, D, D, D, D, D, P, P, P, P, P, P, // 3_
    P, Ux,Ux,Ux,Ux,Ux,Ux,U, U, U, U, U, U, U, U, U, // 4_
    U, U, U, U, U, U, U, U, U, U, U, P, P, P, P, P, // 5_
    P, Lx,Lx,Lx,Lx,Lx,Lx,L, L, L, L, L, L, L, L, L, // 6_
    L, L, L, L, L, L, L, L, L, L, L, P, P, P, P, C, // 7_
    N, N, N, N, N, N, N, N, N, N, N, N, N, N, N, N,
    N, N, N, N, N, N, N, N, N, N, N, N, N, N, N, N,
    N, N, N, N, N, N, N, N, N, N, N, N, N, N, N, N,
    N, N, N, N, N, N, N, N, N, N, N, N, N, N, N, N,
    N, N, N, N, N, N, N, N, N, N, N, N, N, N, N, N,
    N, N, N, N, N, N, N, N, N, N, N, N, N, N, N, N,
    N, N, N, N, N, N, N, N, N, N, N, N, N, N, N, N,
    N, N, N, N, N, N, N, N, N, N, N, N, N, N, N, N,
];

const ASCII_PATH: &[u8] = b"home/kyubey/rust/build/x86_64-unknown-linux-gnu/stage0/lib:/home/kyubey/workspace/rust/build/x86_64-unknown-linux-gnu/stage0-tools/release/deps";
const RUST_INCANTATION: &[u8] = br#"AR_x86_64_unknown_linux_gnu="ar" CARGO_INCREMENTAL="0" CARGO_PROFILE_RELEASE_DEBUG="1" CARGO_PROFILE_RELEASE_DEBUG_ASSERTIONS="false" CARGO_PROFILE_RELEASE_OVERFLOW_CHECKS="false" CARGO_TARGET_DIR="/home/kyubey/workspace/rust/build/x86_64-unknown-linux-gnu/stage0-std" CC_x86_64_unknown_linux_gnu="cc" CFG_COMPILER_HOST_TRIPLE="x86_64-unknown-linux-gnu" CFG_RELEASE_CHANNEL="dev" CFLAGS_x86_64_unknown_linux_gnu="-ffunction-sections -fdata-sections -fPIC -m64" CXXFLAGS_x86_64_unknown_linux_gnu="-ffunction-sections -fdata-sections -fPIC -m64" CXX_x86_64_unknown_linux_gnu="c++" LD_LIBRARY_PATH="/home/kyubey/workspace/rust/build/x86_64-unknown-linux-gnu/stage0-sysroot/lib/rustlib/x86_64-unknown-linux-gnu/lib" LIBC_CHECK_CFG="1" RANLIB_x86_64_unknown_linux_gnu="ar s" REAL_LIBRARY_PATH_VAR="LD_LIBRARY_PATH" RUSTBUILD_NATIVE_DIR="/home/kyubey/workspace/rust/build/x86_64-unknown-linux-gnu/native" RUSTC="/home/kyubey/workspace/rust/build/bootstrap/debug/rustc" RUSTC_BOOTSTRAP="1" RUSTC_BREAK_ON_ICE="1" RUSTC_ERROR_METADATA_DST="/home/kyubey/workspace/rust/build/tmp/extended-error-metadata" RUSTC_FORCE_UNSTABLE="1" RUSTC_HOST_FUSE_LD_LLD="1" RUSTC_INSTALL_BINDIR="bin" RUSTC_LIBDIR="/home/kyubey/workspace/rust/build/x86_64-unknown-linux-gnu/stage0/lib" RUSTC_LINT_FLAGS="-Wrust_2018_idioms -Wunused_lifetimes -Wsemicolon_in_expressions_from_macros" RUSTC_REAL="/home/kyubey/workspace/rust/build/x86_64-unknown-linux-gnu/stage0/bin/rustc" RUSTC_SNAPSHOT="/home/kyubey/workspace/rust/build/x86_64-unknown-linux-gnu/stage0/bin/rustc" RUSTC_SNAPSHOT_LIBDIR="/home/kyubey/workspace/rust/build/x86_64-unknown-linux-gnu/stage0/lib" RUSTC_STAGE="0" RUSTC_SYSROOT="/home/kyubey/workspace/rust/build/x86_64-unknown-linux-gnu/stage0-sysroot" RUSTC_VERBOSE="0" RUSTDOC="/home/kyubey/workspace/rust/build/bootstrap/debug/rustdoc" RUSTDOCFLAGS="-C target-cpu=native --cfg=bootstrap -Csymbol-mangling-version=legacy -Zunstable-options -Zunstable-options --check-cfg=values(bootstrap) --check-cfg=values(stdarch_intel_sde) --check-cfg=values(no_fp_fmt_parse) --check-cfg=values(no_global_oom_handling) --check-cfg=values(no_rc) --check-cfg=values(no_sync) --check-cfg=values(freebsd12) --check-cfg=values(freebsd13) --check-cfg=values(backtrace_in_libstd) --check-cfg=values(target_env,\"libnx\") --check-cfg=values(target_arch,\"asmjs\",\"spirv\",\"nvptx\",\"xtensa\") -Clink-arg=-fuse-ld=lld -Clink-arg=-Wl,--threads=1 -Wrustdoc::invalid_codeblock_attributes --crate-version 1.72.0-dev -Zcrate-attr=doc(html_root_url=\"https://doc.rust-lang.org/nightly/\") -Zcrate-attr=warn(rust_2018_idioms)" RUSTDOC_FUSE_LD_LLD="1" RUSTDOC_LIBDIR="/home/kyubey/workspace/rust/build/x86_64-unknown-linux-gnu/stage0/lib" RUSTDOC_REAL="/path/to/nowhere/rustdoc/not/required" RUSTFLAGS="-C target-cpu=native --cfg=bootstrap -Csymbol-mangling-version=legacy -Zunstable-options -Zunstable-options --check-cfg=values(bootstrap) --check-cfg=values(stdarch_intel_sde) --check-cfg=values(no_fp_fmt_parse) --check-cfg=values(no_global_oom_handling) --check-cfg=values(no_rc) --check-cfg=values(no_sync) --check-cfg=values(freebsd12) --check-cfg=values(freebsd13) --check-cfg=values(backtrace_in_libstd) --check-cfg=values(target_env,\"libnx\") --check-cfg=values(target_arch,\"asmjs\",\"spirv\",\"nvptx\",\"xtensa\") -Zmacro-backtrace -Clink-args=-Wl,-z,origin -Clink-args=-Wl,-rpath,$ORIGIN/../lib -Clink-args=-fuse-ld=lld -Csplit-debuginfo=off -Cprefer-dynamic -Zinline-mir -Clto=off -Zcrate-attr=doc(html_root_url=\"https://doc.rust-lang.org/nightly/\")" RUST_COMPILER_RT_ROOT="/home/kyubey/workspace/rust/src/llvm-project/compiler-rt" RUST_TEST_THREADS="48" WINAPI_NO_BUNDLED_LIBRARIES="1" __CARGO_DEFAULT_LIB_METADATA="bootstrapstd" "/home/kyubey/workspace/rust/build/x86_64-unknown-linux-gnu/stage0/bin/cargo" "bench" "--target" "x86_64-unknown-linux-gnu" "-Zcheck-cfg=names,values,output" "-Zbinary-dep-depinfo" "-j" "48" "--features" " panic-unwind backtrace compiler-builtins-c" "--manifest-path" "/home/kyubey/workspace/rust/library/sysroot/Cargo.toml" "-p" "core" "--" "bench_ascii_escape_display" "--quiet" "-Z" "unstable-options" "--format" "json""#;

#[bench]
fn bench_ascii_escape_display_no_escape(b: &mut Bencher) {
    let mut writer = String::with_capacity(8 * 1024);

    b.iter(move || {
        writer.clear();
        let iter = ASCII_PATH.escape_ascii();
        write!(writer, "{}", iter).unwrap();
        writer.len()
    })
}

#[bench]
fn bench_ascii_escape_display_mixed(b: &mut Bencher) {
    let mut writer = String::with_capacity(8 * 1024);

    b.iter(move || {
        writer.clear();
        let iter = RUST_INCANTATION.escape_ascii();
        write!(writer, "{}", iter).unwrap();
        writer.len()
    })
}
