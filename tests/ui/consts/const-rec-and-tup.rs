// run-pass
#![allow(dead_code)]
#![allow(non_upper_case_globals)]
#![allow(overflowing_literals)]

struct Pair { a: f64, b: f64 }

struct AnotherPair { x: (i64, i64), y: Pair }

static x : (i32,i32) = (0xfeedf00dd,0xca11ab1e);
static y : AnotherPair = AnotherPair{ x: (0xf0f0f0f0_f0f0f0f0,
                                          0xabababab_abababab),
                            y: Pair { a: 3.14159265358979323846,
                                      b: 2.7182818284590452354 }};

pub fn main() {
    let (p, _) = y.x;
    assert_eq!(p, - 1085102592571150096);
    println!("{:#x}", p);
}
