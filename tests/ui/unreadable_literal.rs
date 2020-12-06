// run-rustfix

#![warn(clippy::unreadable_literal)]

struct Foo(u64);

macro_rules! foo {
    () => {
        Foo(123123123123)
    };
}

struct Bar(f32);

macro_rules! bar {
    () => {
        Bar(100200300400.100200300400500)
    };
}

fn main() {
    let _good = (
        0b1011_i64,
        0o1_234_u32,
        0x1_234_567,
        65536,
        1_2345_6789,
        1234_f32,
        1_234.12_f32,
        1_234.123_f32,
        1.123_4_f32,
    );
    let _bad = (0b110110_i64, 0xcafebabe_usize, 123456_f32, 1.234567_f32);
    let _good_sci = 1.1234e1;
    let _bad_sci = 1.123456e1;

    let _fail1 = 0xabcdef;
    let _fail2: u32 = 0xBAFEBAFE;
    let _fail3 = 0xabcdeff;
    let _fail4: i128 = 0xabcabcabcabcabcabc;
    let _fail5 = 1.100300400;

    let _ = foo!();
    let _ = bar!();
}
