#![warn(clippy::unnecessary_cast)]
#![allow(clippy::no_effect)]

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
}
