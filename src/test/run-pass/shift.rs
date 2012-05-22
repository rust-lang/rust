// Testing shifts for various combinations of integers
// Issue #1570

fn main() {
    test_misc();
    test_expr();
    test_const();
}

fn test_misc() {
    assert 1 << 1i8 << 1u8 << 1i16 << 1 as char << 1u64 == 32;
}

fn test_expr() {
    let v10 = 10 as uint;
    let v4 = 4 as u8;
    let v2 = 2 as u8;
    assert (v10 >> v2 == v2 as uint);
    assert (v10 << v4 == 160 as uint);

    let v10 = 10 as u8;
    let v4 = 4 as uint;
    let v2 = 2 as uint;
    assert (v10 >> v2 == v2 as u8);
    assert (v10 << v4 == 160 as u8);

    let v10 = 10 as int;
    let v4 = 4 as i8;
    let v2 = 2 as i8;
    assert (v10 >> v2 == v2 as int);
    assert (v10 << v4 == 160 as int);

    let v10 = 10 as i8;
    let v4 = 4 as int;
    let v2 = 2 as int;
    assert (v10 >> v2 == v2 as i8);
    assert (v10 << v4 == 160 as i8);

    let v10 = 10 as uint;
    let v4 = 4 as int;
    let v2 = 2 as int;
    assert (v10 >> v2 == v2 as uint);
    assert (v10 << v4 == 160 as uint);
}

fn test_const() {
    const r1_1: uint = 10u >> 2u8;
    const r2_1: uint = 10u << 4u8;
    assert r1_1 == 2 as uint;
    assert r2_1 == 160 as uint;

    const r1_2: u8 = 10u8 >> 2u;
    const r2_2: u8 = 10u8 << 4u;
    assert r1_2 == 2 as u8;
    assert r2_2 == 160 as u8;

    const r1_3: int = 10 >> 2i8;
    const r2_3: int = 10 << 4i8;
    assert r1_3 == 2 as int;
    assert r2_3 == 160 as int;

    const r1_4: i8 = 10i8 >> 2;
    const r2_4: i8 = 10i8 << 4;
    assert r1_4 == 2 as i8;
    assert r2_4 == 160 as i8;

    const r1_5: uint = 10u >> 2i8;
    const r2_5: uint = 10u << 4i8;
    assert r1_5 == 2 as uint;
    assert r2_5 == 160 as uint;
}
