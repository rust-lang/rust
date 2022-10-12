fn ret() -> i64 {
    1
}

fn neg() -> i64 {
    -1
}

fn add() -> i64 {
    1 + 2
}

fn indirect_add() -> i64 {
    let x = 1;
    let y = 2;
    x + y
}

fn arith() -> i32 {
    3 * 3 + 4 * 4
}

fn match_int() -> i16 {
    let n = 2;
    match n {
        0 => 0,
        1 => 10,
        2 => 20,
        3 => 30,
        _ => 100,
    }
}

fn match_int_range() -> i64 {
    let n = 42;
    match n {
        0..=9 => 0,
        10..=19 => 1,
        20..=29 => 2,
        30..=39 => 3,
        40..=42 => 4,
        _ => 5,
    }
}

fn main() {
    assert_eq!(ret(), 1);
    assert_eq!(neg(), -1);
    assert_eq!(add(), 3);
    assert_eq!(indirect_add(), 3);
    assert_eq!(arith(), 5 * 5);
    assert_eq!(match_int(), 20);
    assert_eq!(match_int_range(), 4);
    assert_eq!(i64::MIN.overflowing_mul(-1), (i64::MIN, true));
    assert_eq!(i32::MIN.overflowing_mul(-1), (i32::MIN, true));
    assert_eq!(i16::MIN.overflowing_mul(-1), (i16::MIN, true));
    assert_eq!(i8::MIN.overflowing_mul(-1), (i8::MIN, true));
}
