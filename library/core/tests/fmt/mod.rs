mod builders;
mod float;
mod num;

use core::any::Any;
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
    let b: &[u8] = b"";
    let s: &str = "";
    assert_eq!(format!("{:p}", s), format!("{:p}", s.as_ptr()));
    assert_eq!(format!("{:p}", b), format!("{:p}", b.as_ptr()));
}

#[test]
fn test_pointer_formats_debug_thin() {
    let thinptr = &42 as *const i32;
    assert_eq!(format!("{:?}", thinptr as *const ()), format!("{:p}", thinptr));
}

#[test]
fn test_pointer_formats_debug_slice() {
    let b: &[u8] = b"hello";
    let s: &str = "hello";
    let b_ptr = &*b as *const _;
    let s_ptr = &*s as *const _;
    assert_eq!(format!("{:?}", b_ptr), format!("({:?}, 5)", b.as_ptr()));
    assert_eq!(format!("{:?}", s_ptr), format!("({:?}, 5)", s.as_ptr()));

    // :p should format as a thin pointer / without metadata
    assert_eq!(format!("{:p}", b_ptr), format!("{:p}", b.as_ptr()));
    assert_eq!(format!("{:p}", s_ptr), format!("{:p}", s.as_ptr()));
}

#[test]
fn test_pointer_formats_debug_trait_object() {
    let mut any: Box<dyn Any> = Box::new(42);
    let dyn_ptr = &mut *any as *mut dyn Any;
    assert_eq!(format!("{:?}", dyn_ptr), format!("({:?}, {:?})", dyn_ptr as *const (), ptr::metadata(dyn_ptr)));

    // :p should format as a thin pointer / without metadata
    assert_eq!(format!("{:p}", dyn_ptr), format!("{:p}", dyn_ptr as *const ()));
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
