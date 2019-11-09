#![feature(ptr_offset_from)]

fn test_raw() { unsafe {
    let buf = [0u32; 4];

    let x = buf.as_ptr() as *const u8;
    let y = x.offset(12);

    assert_eq!(y.offset_from(x), 12);
    assert_eq!(x.offset_from(y), -12);
    assert_eq!((y as *const u32).offset_from(x as *const u32), 12/4);
    assert_eq!((x as *const u32).offset_from(y as *const u32), -12/4);
    
    let x = (((x as usize) * 2) / 2) as *const u8;
    assert_eq!(y.offset_from(x), 12);
    assert_eq!(x.offset_from(y), -12);
} }

// This also internally uses offset_from.
fn test_vec_into_iter() {
    let v = Vec::<i32>::new();
    let i = v.into_iter();
    i.size_hint();
}

fn main() {
    test_raw();
    test_vec_into_iter();
}
