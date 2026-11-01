//@ run-pass
// Test that we can do shifts by any integral type.


struct Panolpy {
    i8: i8,
    i16: i16,
    i32: i32,
    i64: i64,
    isize: isize,

    u8: u8,
    u16: u16,
    u32: u32,
    u64: u64,
    usize: usize,
}

fn foo(p: &Panolpy) {
    assert_eq!(22 >> p.i8, 11);
    assert_eq!(22 >> p.i16, 11);
    assert_eq!(22 >> p.i32, 11);
    assert_eq!(22 >> p.i64, 11);
    assert_eq!(22 >> p.isize, 11);

    assert_eq!(22 >> p.u8, 11);
    assert_eq!(22 >> p.u16, 11);
    assert_eq!(22 >> p.u32, 11);
    assert_eq!(22 >> p.u64, 11);
    assert_eq!(22 >> p.usize, 11);
}

fn main() {
    let p = Panolpy {
        i8: 1,
        i16: 1,
        i32: 1,
        i64: 1,
        isize: 1,

        u8: 1,
        u16: 1,
        u32: 1,
        u64: 1,
        usize: 1,
    };
    foo(&p)
}
