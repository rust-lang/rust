mod builders;
mod float;
mod num;

use core::any;
use core::fmt;
use core::ptr;

#[test]
fn test_format_flags() {
    // No residual flags left by pointer formatting
    let p = "".as_ptr();
    assert_eq!(format!("{:p} {:x}", p, 16), format!("{:p} 10", p));

    assert_eq!(format!("{: >3}", 'a'), "  a");
}

#[test]
fn test_pointer_formats_data_pointer() {
    // Thin ptr
    let thinptr = &42 as *const i32;
    assert_eq!(format!("{:p}", thinptr), format!("{:?}", thinptr as *const ()));

    // Ptr with length
    let b: &[u8] = b"hello";
    let s: &str = "hello";
    assert_eq!(format!("{:p}", b), format!("({:?}, 5)", b.as_ptr()));
    assert_eq!(format!("{:p}", s), format!("({:?}, 5)", s.as_ptr()));

    // Ptr with v-table
    let mut any: Box<dyn any::Any> = Box::new(42);
    let dyn_ptr = &mut *any as *mut dyn any::Any;
    assert_eq!(format!("{:p}", dyn_ptr), format!("({:?}, {:?})", dyn_ptr as *const (), ptr::metadata(dyn_ptr)));
}

#[test]
fn test_estimated_capacity() {
    assert_eq!(format_args!("").estimated_capacity(), 0);
    assert_eq!(format_args!("{}", "").estimated_capacity(), 0);
    assert_eq!(format_args!("Hello").estimated_capacity(), 5);
    assert_eq!(format_args!("Hello, {}!", "").estimated_capacity(), 16);
    assert_eq!(format_args!("{}, hello!", "World").estimated_capacity(), 0);
    assert_eq!(format_args!("{}. 16-bytes piece", "World").estimated_capacity(), 32);
}

#[test]
fn pad_integral_resets() {
    struct Bar;

    impl fmt::Display for Bar {
        fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
            "1".fmt(f)?;
            f.pad_integral(true, "", "5")?;
            "1".fmt(f)
        }
    }

    assert_eq!(format!("{:<03}", Bar), "1  0051  ");
}
