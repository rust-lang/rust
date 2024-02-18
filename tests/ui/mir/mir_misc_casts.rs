//@ run-pass
fn func(){}

const STR: &'static str = "hello";
const BSTR: &'static [u8; 5] = b"hello";

fn from_ptr()
-> (isize, usize, i8, i16, i32, i64, u8, u16, u32, u64, *const ()) {
    let f = 1_usize as *const String;
    let c1 = f as isize;
    let c2 = f as usize;
    let c3 = f as i8;
    let c4 = f as i16;
    let c5 = f as i32;
    let c6 = f as i64;
    let c7 = f as u8;
    let c8 = f as u16;
    let c9 = f as u32;
    let c10 = f as u64;
    let c11 = f as *const ();
    (c1, c2, c3, c4, c5, c6, c7, c8, c9, c10, c11)
}

fn from_1()
-> (isize, usize, i8, i16, i32, i64, u8, u16, u32, u64, f32, f64, *const String) {
    let c1 = 1 as isize;
    let c2 = 1 as usize;
    let c3 = 1 as i8;
    let c4 = 1 as i16;
    let c5 = 1 as i32;
    let c6 = 1 as i64;
    let c7 = 1 as u8;
    let c8 = 1 as u16;
    let c9 = 1 as u32;
    let c10 = 1 as u64;
    let c11 = 1 as f32;
    let c12 = 1 as f64;
    let c13 = 1 as *const String;
    (c1, c2, c3, c4, c5, c6, c7, c8, c9, c10, c11, c12, c13)
}

fn from_1usize()
-> (isize, usize, i8, i16, i32, i64, u8, u16, u32, u64, f32, f64, *const String) {
    let c1 = 1_usize as isize;
    let c2 = 1_usize as usize;
    let c3 = 1_usize as i8;
    let c4 = 1_usize as i16;
    let c5 = 1_usize as i32;
    let c6 = 1_usize as i64;
    let c7 = 1_usize as u8;
    let c8 = 1_usize as u16;
    let c9 = 1_usize as u32;
    let c10 = 1_usize as u64;
    let c11 = 1_usize as f32;
    let c12 = 1_usize as f64;
    let c13 = 1_usize as *const String;
    (c1, c2, c3, c4, c5, c6, c7, c8, c9, c10, c11, c12, c13)
}

fn from_1isize()
-> (isize, usize, i8, i16, i32, i64, u8, u16, u32, u64, f32, f64, *const String) {
    let c1 = 1_isize as isize;
    let c2 = 1_isize as usize;
    let c3 = 1_isize as i8;
    let c4 = 1_isize as i16;
    let c5 = 1_isize as i32;
    let c6 = 1_isize as i64;
    let c7 = 1_isize as u8;
    let c8 = 1_isize as u16;
    let c9 = 1_isize as u32;
    let c10 = 1_isize as u64;
    let c11 = 1_isize as f32;
    let c12 = 1_isize as f64;
    let c13 = 1_isize as *const String;
    (c1, c2, c3, c4, c5, c6, c7, c8, c9, c10, c11, c12, c13)
}

fn from_1u8()
-> (isize, usize, i8, i16, i32, i64, u8, u16, u32, u64, f32, f64, *const String) {
    let c1 = 1_u8 as isize;
    let c2 = 1_u8 as usize;
    let c3 = 1_u8 as i8;
    let c4 = 1_u8 as i16;
    let c5 = 1_u8 as i32;
    let c6 = 1_u8 as i64;
    let c7 = 1_u8 as u8;
    let c8 = 1_u8 as u16;
    let c9 = 1_u8 as u32;
    let c10 = 1_u8 as u64;
    let c11 = 1_u8 as f32;
    let c12 = 1_u8 as f64;
    let c13 = 1_u8 as *const String;
    (c1, c2, c3, c4, c5, c6, c7, c8, c9, c10, c11, c12, c13)
}

fn from_1i8()
-> (isize, usize, i8, i16, i32, i64, u8, u16, u32, u64, f32, f64, *const String) {
    let c1 = 1_i8 as isize;
    let c2 = 1_i8 as usize;
    let c3 = 1_i8 as i8;
    let c4 = 1_i8 as i16;
    let c5 = 1_i8 as i32;
    let c6 = 1_i8 as i64;
    let c7 = 1_i8 as u8;
    let c8 = 1_i8 as u16;
    let c9 = 1_i8 as u32;
    let c10 = 1_i8 as u64;
    let c11 = 1_i8 as f32;
    let c12 = 1_i8 as f64;
    let c13 = 1_i8 as *const String;
    (c1, c2, c3, c4, c5, c6, c7, c8, c9, c10, c11, c12, c13)
}

fn from_1u16()
-> (isize, usize, i8, i16, i32, i64, u8, u16, u32, u64, f32, f64, *const String) {
    let c1 = 1_u16 as isize;
    let c2 = 1_u16 as usize;
    let c3 = 1_u16 as i8;
    let c4 = 1_u16 as i16;
    let c5 = 1_u16 as i32;
    let c6 = 1_u16 as i64;
    let c7 = 1_u16 as u8;
    let c8 = 1_u16 as u16;
    let c9 = 1_u16 as u32;
    let c10 = 1_u16 as u64;
    let c11 = 1_u16 as f32;
    let c12 = 1_u16 as f64;
    let c13 = 1_u16 as *const String;
    (c1, c2, c3, c4, c5, c6, c7, c8, c9, c10, c11, c12, c13)
}

fn from_1i16()
-> (isize, usize, i8, i16, i32, i64, u8, u16, u32, u64, f32, f64, *const String) {
    let c1 = 1_i16 as isize;
    let c2 = 1_i16 as usize;
    let c3 = 1_i16 as i8;
    let c4 = 1_i16 as i16;
    let c5 = 1_i16 as i32;
    let c6 = 1_i16 as i64;
    let c7 = 1_i16 as u8;
    let c8 = 1_i16 as u16;
    let c9 = 1_i16 as u32;
    let c10 = 1_i16 as u64;
    let c11 = 1_i16 as f32;
    let c12 = 1_i16 as f64;
    let c13 = 1_i16 as *const String;
    (c1, c2, c3, c4, c5, c6, c7, c8, c9, c10, c11, c12, c13)
}

fn from_1u32()
-> (isize, usize, i8, i16, i32, i64, u8, u16, u32, u64, f32, f64, *const String) {
    let c1 = 1_u32 as isize;
    let c2 = 1_u32 as usize;
    let c3 = 1_u32 as i8;
    let c4 = 1_u32 as i16;
    let c5 = 1_u32 as i32;
    let c6 = 1_u32 as i64;
    let c7 = 1_u32 as u8;
    let c8 = 1_u32 as u16;
    let c9 = 1_u32 as u32;
    let c10 = 1_u32 as u64;
    let c11 = 1_u32 as f32;
    let c12 = 1_u32 as f64;
    let c13 = 1_u32 as *const String;
    (c1, c2, c3, c4, c5, c6, c7, c8, c9, c10, c11, c12, c13)
}

fn from_1i32()
-> (isize, usize, i8, i16, i32, i64, u8, u16, u32, u64, f32, f64, *const String) {
    let c1 = 1_i32 as isize;
    let c2 = 1_i32 as usize;
    let c3 = 1_i32 as i8;
    let c4 = 1_i32 as i16;
    let c5 = 1_i32 as i32;
    let c6 = 1_i32 as i64;
    let c7 = 1_i32 as u8;
    let c8 = 1_i32 as u16;
    let c9 = 1_i32 as u32;
    let c10 = 1_i32 as u64;
    let c11 = 1_i32 as f32;
    let c12 = 1_i32 as f64;
    let c13 = 1_i32 as *const String;
    (c1, c2, c3, c4, c5, c6, c7, c8, c9, c10, c11, c12, c13)
}

fn from_1u64()
-> (isize, usize, i8, i16, i32, i64, u8, u16, u32, u64, f32, f64, *const String) {
    let c1 = 1_u64 as isize;
    let c2 = 1_u64 as usize;
    let c3 = 1_u64 as i8;
    let c4 = 1_u64 as i16;
    let c5 = 1_u64 as i32;
    let c6 = 1_u64 as i64;
    let c7 = 1_u64 as u8;
    let c8 = 1_u64 as u16;
    let c9 = 1_u64 as u32;
    let c10 = 1_u64 as u64;
    let c11 = 1_u64 as f32;
    let c12 = 1_u64 as f64;
    let c13 = 1_u64 as *const String;
    (c1, c2, c3, c4, c5, c6, c7, c8, c9, c10, c11, c12, c13)
}

fn from_1i64()
-> (isize, usize, i8, i16, i32, i64, u8, u16, u32, u64, f32, f64, *const String) {
    let c1 = 1_i64 as isize;
    let c2 = 1_i64 as usize;
    let c3 = 1_i64 as i8;
    let c4 = 1_i64 as i16;
    let c5 = 1_i64 as i32;
    let c6 = 1_i64 as i64;
    let c7 = 1_i64 as u8;
    let c8 = 1_i64 as u16;
    let c9 = 1_i64 as u32;
    let c10 = 1_i64 as u64;
    let c11 = 1_i64 as f32;
    let c12 = 1_i64 as f64;
    let c13 = 1_i64 as *const String;
    (c1, c2, c3, c4, c5, c6, c7, c8, c9, c10, c11, c12, c13)
}

fn from_bool()
-> (isize, usize, i8, i16, i32, i64, u8, u16, u32, u64) {
    let c1 = true as isize;
    let c2 = true as usize;
    let c3 = true as i8;
    let c4 = true as i16;
    let c5 = true as i32;
    let c6 = true as i64;
    let c7 = true as u8;
    let c8 = true as u16;
    let c9 = true as u32;
    let c10 = true as u64;
    (c1, c2, c3, c4, c5, c6, c7, c8, c9, c10)
}

fn from_1f32()
-> (isize, usize, i8, i16, i32, i64, u8, u16, u32, u64, f32, f64) {
    let c1 = 1.0_f32 as isize;
    let c2 = 1.0_f32 as usize;
    let c3 = 1.0_f32 as i8;
    let c4 = 1.0_f32 as i16;
    let c5 = 1.0_f32 as i32;
    let c6 = 1.0_f32 as i64;
    let c7 = 1.0_f32 as u8;
    let c8 = 1.0_f32 as u16;
    let c9 = 1.0_f32 as u32;
    let c10 = 1.0_f32 as u64;
    let c11 = 1.0_f32 as f32;
    let c12 = 1.0_f32 as f64;
    (c1, c2, c3, c4, c5, c6, c7, c8, c9, c10, c11, c12)
}

fn from_1f64()
-> (isize, usize, i8, i16, i32, i64, u8, u16, u32, u64, f32, f64) {
    let c1 = 1.0f64 as isize;
    let c2 = 1.0f64 as usize;
    let c3 = 1.0f64 as i8;
    let c4 = 1.0f64 as i16;
    let c5 = 1.0f64 as i32;
    let c6 = 1.0f64 as i64;
    let c7 = 1.0f64 as u8;
    let c8 = 1.0f64 as u16;
    let c9 = 1.0f64 as u32;
    let c10 = 1.0f64 as u64;
    let c11 = 1.0f64 as f32;
    let c12 = 1.0f64 as f64;
    (c1, c2, c3, c4, c5, c6, c7, c8, c9, c10, c11, c12)
}

fn other_casts()
-> (*const u8, *const isize, *const u8, *const u8) {
    let c1 = func as *const u8;
    let c2 = c1 as *const isize;

    let r = &42u32;
    let _ = r as *const u32;

    // fat-ptr -> fat-ptr -> fat-raw-ptr -> thin-ptr
    let c3 = STR as &str as *const str as *const u8;

    let c4 = BSTR as *const [u8] as *const [u16] as *const u8;
    (c1, c2, c3, c4)
}

pub fn assert_eq_13(l: (isize, usize, i8, i16, i32, i64, u8,
                        u16, u32, u64, f32, f64, *const String),
                    r: (isize, usize, i8, i16, i32, i64, u8,
                        u16, u32, u64, f32, f64, *const String)) -> bool {
    let (l1, l2, l3, l4, l5, l6, l7, l8, l9, l10, l11, l12, l13) = l;
    let (r1, r2, r3, r4, r5, r6, r7, r8, r9, r10, r11, r12, r13) = r;
    l1 == r1 && l2 == r2 && l3 == r3 && l4 == r4 && l5 == r5 && l6 == r6 && l7 == r7 &&
    l8 == r8 && l9 == r9 && l10 == r10 && l11 == r11 && l12 == r12 && l13 == r13
}


pub fn main() {
    let f = 1_usize as *const String;
    let t13 = (1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1.0, 1.0, f);
    let t12 = (1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1.0, 1.0);
    assert_eq_13(from_1(), t13);
    assert_eq_13(from_1usize(), t13);
    assert_eq_13(from_1isize(), t13);
    assert_eq_13(from_1u8(), t13);
    assert_eq_13(from_1i8(), t13);
    assert_eq_13(from_1u16(), t13);
    assert_eq_13(from_1i16(), t13);
    assert_eq_13(from_1u32(), t13);
    assert_eq_13(from_1i32(), t13);
    assert_eq_13(from_1u64(), t13);
    assert_eq_13(from_1i64(), t13);
    assert_eq!(from_1f32(), t12);
    assert_eq!(from_1f64(), t12);

    assert_eq!(from_ptr(), (1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1 as *const ()));
    assert_eq!(from_bool(), (1, 1, 1, 1, 1, 1, 1, 1, 1, 1));

    assert_eq!(other_casts(), (func as *const u8, func as *const isize,
                               STR as *const str as *const u8, BSTR as *const [u8] as *const u8));
}
