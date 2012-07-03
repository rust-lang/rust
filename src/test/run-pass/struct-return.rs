type quad = { a: u64, b: u64, c: u64, d: u64 };
type floats = { a: f64, b: u8, c: f64 };

#[nolink]
extern mod rustrt {
    fn debug_abi_1(++q: quad) -> quad;
    fn debug_abi_2(++f: floats) -> floats;
}

fn test1() {
    let q = { a: 0xaaaa_aaaa_aaaa_aaaa_u64,
             b: 0xbbbb_bbbb_bbbb_bbbb_u64,
             c: 0xcccc_cccc_cccc_cccc_u64,
             d: 0xdddd_dddd_dddd_dddd_u64 };
    let qq = rustrt::debug_abi_1(q);
    #error("a: %x", qq.a as uint);
    #error("b: %x", qq.b as uint);
    #error("c: %x", qq.c as uint);
    #error("d: %x", qq.d as uint);
    assert qq.a == q.c + 1u64;
    assert qq.b == q.d - 1u64;
    assert qq.c == q.a + 1u64;
    assert qq.d == q.b - 1u64;
}

#[cfg(target_arch = "x86_64")]
fn test2() {
    let f = { a: 1.234567890e-15_f64,
             b: 0b_1010_1010_u8,
             c: 1.0987654321e-15_f64 };
    let ff = rustrt::debug_abi_2(f);
    #error("a: %f", ff.a as float);
    #error("b: %u", ff.b as uint);
    #error("c: %f", ff.c as float);
    assert ff.a == f.c + 1.0f64;
    assert ff.b == 0xff_u8;
    assert ff.c == f.a - 1.0f64;
}

#[cfg(target_arch = "x86")]
fn test2() {
}

fn main() {
    test1();
    test2();
}