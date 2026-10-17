//@ run-pass
//@ reference: layout.aggregate.struct-offsets
//@ edition: 2018

#[repr(align(64))]
#[derive(Copy, Clone)]
#[allow(dead_code)]
pub struct Overaligned(u8);

#[allow(dead_code)]
struct ReprRustStruct {
    x: i32,
    y: [u32; 4],
    z: f32,
    a: u128,
    b: Overaligned,
}

macro_rules! span_of {
    ($ty:ty , $field:tt) => {{
        let __field = unsafe { ::core::mem::zeroed::<$ty>() };

        (
            core::mem::offset_of!($ty, $field),
            core::mem::offset_of!($ty, $field) + core::mem::size_of_val(&__field.$field),
        )
    }};
}

fn test_fields_make_sense(a: &(usize, usize)) {
    assert!(a.0 <= a.1);
}

// order is `begin, end`
fn test_non_overlapping(a: &(usize, usize), b: &(usize, usize)) {
    assert!((a.1 <= b.0) || (b.1 <= a.0));
}

fn test_fields_non_overlapping() {
    let fields = [
        span_of!(ReprRustStruct, x),
        span_of!(ReprRustStruct, y),
        span_of!(ReprRustStruct, z),
        span_of!(ReprRustStruct, a),
        span_of!(ReprRustStruct, b),
    ];

    test_fields_make_sense(&fields[0]);
    test_fields_make_sense(&fields[1]);
    test_fields_make_sense(&fields[2]);
    test_fields_make_sense(&fields[3]);
    test_fields_make_sense(&fields[4]);

    test_non_overlapping(&fields[0], &fields[1]);
    test_non_overlapping(&fields[0], &fields[2]);
    test_non_overlapping(&fields[0], &fields[3]);
    test_non_overlapping(&fields[0], &fields[4]);
    test_non_overlapping(&fields[1], &fields[2]);
    test_non_overlapping(&fields[2], &fields[3]);
    test_non_overlapping(&fields[2], &fields[4]);
    test_non_overlapping(&fields[3], &fields[4]);
}

fn test_fields_aligned() {
    assert_eq!((core::mem::offset_of!(ReprRustStruct, x) % (core::mem::align_of::<i32>())), 0);
    assert_eq!((core::mem::offset_of!(ReprRustStruct, y) % (core::mem::align_of::<[u32; 4]>())), 0);
    assert_eq!((core::mem::offset_of!(ReprRustStruct, z) % (core::mem::align_of::<f32>())), 0);
    assert_eq!((core::mem::offset_of!(ReprRustStruct, a) % (core::mem::align_of::<u128>())), 0);
    assert_eq!(
        (core::mem::offset_of!(ReprRustStruct, b) % (core::mem::align_of::<Overaligned>())),
        0
    );
}

fn main() {
    test_fields_non_overlapping();
    test_fields_aligned();
}
