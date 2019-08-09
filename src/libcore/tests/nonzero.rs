use core::num::{IntErrorKind, NonZeroI32, NonZeroI8, NonZeroU32, NonZeroU8};
use core::option::Option::{self, None, Some};
use std::mem::size_of;

#[test]
fn test_create_nonzero_instance() {
    let _a = unsafe {
        NonZeroU32::new_unchecked(21)
    };
}

#[test]
fn test_size_nonzero_in_option() {
    assert_eq!(size_of::<NonZeroU32>(), size_of::<Option<NonZeroU32>>());
    assert_eq!(size_of::<NonZeroI32>(), size_of::<Option<NonZeroI32>>());
}

#[test]
fn test_match_on_nonzero_option() {
    let a = Some(unsafe {
        NonZeroU32::new_unchecked(42)
    });
    match a {
        Some(val) => assert_eq!(val.get(), 42),
        None => panic!("unexpected None while matching on Some(NonZeroU32(_))")
    }

    match unsafe { Some(NonZeroU32::new_unchecked(43)) } {
        Some(val) => assert_eq!(val.get(), 43),
        None => panic!("unexpected None while matching on Some(NonZeroU32(_))")
    }
}

#[test]
fn test_match_option_empty_vec() {
    let a: Option<Vec<isize>> = Some(vec![]);
    match a {
        None => panic!("unexpected None while matching on Some(vec![])"),
        _ => {}
    }
}

#[test]
fn test_match_option_vec() {
    let a = Some(vec![1, 2, 3, 4]);
    match a {
        Some(v) => assert_eq!(v, [1, 2, 3, 4]),
        None => panic!("unexpected None while matching on Some(vec![1, 2, 3, 4])")
    }
}

#[test]
fn test_match_option_rc() {
    use std::rc::Rc;

    let five = Rc::new(5);
    match Some(five) {
        Some(r) => assert_eq!(*r, 5),
        None => panic!("unexpected None while matching on Some(Rc::new(5))")
    }
}

#[test]
fn test_match_option_arc() {
    use std::sync::Arc;

    let five = Arc::new(5);
    match Some(five) {
        Some(a) => assert_eq!(*a, 5),
        None => panic!("unexpected None while matching on Some(Arc::new(5))")
    }
}

#[test]
fn test_match_option_empty_string() {
    let a = Some(String::new());
    match a {
        None => panic!("unexpected None while matching on Some(String::new())"),
        _ => {}
    }
}

#[test]
fn test_match_option_string() {
    let five = "Five".to_string();
    match Some(five) {
        Some(s) => assert_eq!(s, "Five"),
        None => panic!("unexpected None while matching on Some(String { ... })")
    }
}

mod atom {
    use core::num::NonZeroU32;

    #[derive(PartialEq, Eq)]
    pub struct Atom {
        index: NonZeroU32, // private
    }
    pub const FOO_ATOM: Atom = Atom { index: unsafe { NonZeroU32::new_unchecked(7) } };
}

macro_rules! atom {
    ("foo") => { atom::FOO_ATOM }
}

#[test]
fn test_match_nonzero_const_pattern() {
    match atom!("foo") {
        // Using as a pattern is supported by the compiler:
        atom!("foo") => {}
        _ => panic!("Expected the const item as a pattern to match.")
    }
}

#[test]
fn test_from_nonzero() {
    let nz = NonZeroU32::new(1).unwrap();
    let num: u32 = nz.into();
    assert_eq!(num, 1u32);
}

#[test]
fn test_from_signed_nonzero() {
    let nz = NonZeroI32::new(1).unwrap();
    let num: i32 = nz.into();
    assert_eq!(num, 1i32);
}

#[test]
fn test_from_str() {
    assert_eq!("123".parse::<NonZeroU8>(), Ok(NonZeroU8::new(123).unwrap()));
    assert_eq!(
        "0".parse::<NonZeroU8>().err().map(|e| e.kind().clone()),
        Some(IntErrorKind::Zero)
    );
    assert_eq!(
        "-1".parse::<NonZeroU8>().err().map(|e| e.kind().clone()),
        Some(IntErrorKind::InvalidDigit)
    );
    assert_eq!(
        "-129".parse::<NonZeroI8>().err().map(|e| e.kind().clone()),
        Some(IntErrorKind::Underflow)
    );
    assert_eq!(
        "257".parse::<NonZeroU8>().err().map(|e| e.kind().clone()),
        Some(IntErrorKind::Overflow)
    );
}
