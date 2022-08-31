// run-rustfix
#![warn(clippy::unnecessary_cast)]
#![allow(
    unused_must_use,
    clippy::borrow_as_ptr,
    clippy::no_effect,
    clippy::nonstandard_macro_braces,
    clippy::unnecessary_operation
)]

#[rustfmt::skip]
fn main() {
    // Test cast_unnecessary
    1i32 as i32;
    1f32 as f32;
    false as bool;
    &1i32 as &i32;

    -1_i32 as i32;
    - 1_i32 as i32;
    -1f32 as f32;
    1_i32 as i32;
    1_f32 as f32;

    // macro version
    macro_rules! foo {
        ($a:ident, $b:ident) => {
            #[allow(unused)]
            pub fn $a() -> $b {
                1 as $b
            }
        };
    }
    foo!(a, i32);
    foo!(b, f32);
    foo!(c, f64);

    // do not lint cast to cfg-dependant type
    1 as std::os::raw::c_char;

    // do not lint cast to alias type
    1 as I32Alias;
    &1 as &I32Alias;
}

type I32Alias = i32;

mod fixable {
    #![allow(dead_code)]

    fn main() {
        // casting integer literal to float is unnecessary
        100 as f32;
        100 as f64;
        100_i32 as f64;
        let _ = -100 as f32;
        let _ = -100 as f64;
        let _ = -100_i32 as f64;
        100. as f32;
        100. as f64;
        // Should not trigger
        #[rustfmt::skip]
        let v = vec!(1);
        &v as &[i32];
        0x10 as f32;
        0o10 as f32;
        0b10 as f32;
        0x11 as f64;
        0o11 as f64;
        0b11 as f64;

        1 as u32;
        0x10 as i32;
        0b10 as usize;
        0o73 as u16;
        1_000_000_000 as u32;

        1.0 as f64;
        0.5 as f32;

        1.0 as u16;

        let _ = -1 as i32;
        let _ = -1.0 as f32;

        let _ = 1 as I32Alias;
        let _ = &1 as &I32Alias;
    }

    type I32Alias = i32;

    fn issue_9380() {
        let _: i32 = -(1) as i32;
        let _: f32 = -(1) as f32;
        let _: i64 = -(1) as i64;
        let _: i64 = -(1.0) as i64;

        let _ = -(1 + 1) as i64;
    }
}
