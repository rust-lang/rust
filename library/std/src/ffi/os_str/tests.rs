use super::*;
use crate::mem::MaybeUninit;
use crate::ptr;

#[test]
fn test_os_string_with_capacity() {
    let os_string = OsString::with_capacity(0);
    assert_eq!(0, os_string.inner.into_inner().capacity());

    let os_string = OsString::with_capacity(10);
    assert_eq!(10, os_string.inner.into_inner().capacity());

    let mut os_string = OsString::with_capacity(0);
    os_string.push("abc");
    assert!(os_string.inner.into_inner().capacity() >= 3);
}

#[test]
fn test_os_string_clear() {
    let mut os_string = OsString::from("abc");
    assert_eq!(3, os_string.inner.as_inner().len());

    os_string.clear();
    assert_eq!(&os_string, "");
    assert_eq!(0, os_string.inner.as_inner().len());
}

#[test]
fn test_os_string_leak() {
    let os_string = OsString::from("have a cake");
    let (len, cap) = (os_string.len(), os_string.capacity());
    let leaked = os_string.leak();
    assert_eq!(leaked.as_encoded_bytes(), b"have a cake");
    unsafe { drop(String::from_raw_parts(leaked as *mut OsStr as _, len, cap)) }
}

#[test]
fn test_os_string_capacity() {
    let os_string = OsString::with_capacity(0);
    assert_eq!(0, os_string.capacity());

    let os_string = OsString::with_capacity(10);
    assert_eq!(10, os_string.capacity());

    let mut os_string = OsString::with_capacity(0);
    os_string.push("abc");
    assert!(os_string.capacity() >= 3);
}

#[test]
fn test_os_string_reserve() {
    let mut os_string = OsString::new();
    assert_eq!(os_string.capacity(), 0);

    os_string.reserve(2);
    assert!(os_string.capacity() >= 2);

    for _ in 0..16 {
        os_string.push("a");
    }

    assert!(os_string.capacity() >= 16);
    os_string.reserve(16);
    assert!(os_string.capacity() >= 32);

    os_string.push("a");

    os_string.reserve(16);
    assert!(os_string.capacity() >= 33)
}

#[test]
fn test_os_string_reserve_exact() {
    let mut os_string = OsString::new();
    assert_eq!(os_string.capacity(), 0);

    os_string.reserve_exact(2);
    assert!(os_string.capacity() >= 2);

    for _ in 0..16 {
        os_string.push("a");
    }

    assert!(os_string.capacity() >= 16);
    os_string.reserve_exact(16);
    assert!(os_string.capacity() >= 32);

    os_string.push("a");

    os_string.reserve_exact(16);
    assert!(os_string.capacity() >= 33)
}

#[test]
fn test_os_string_join() {
    let strings = [OsStr::new("hello"), OsStr::new("dear"), OsStr::new("world")];
    assert_eq!("hello", strings[..1].join(OsStr::new(" ")));
    assert_eq!("hello dear world", strings.join(OsStr::new(" ")));
    assert_eq!("hellodearworld", strings.join(OsStr::new("")));
    assert_eq!("hello.\n dear.\n world", strings.join(OsStr::new(".\n ")));

    assert_eq!("dear world", strings[1..].join(&OsString::from(" ")));

    let strings_abc = [OsString::from("a"), OsString::from("b"), OsString::from("c")];
    assert_eq!("a b c", strings_abc.join(OsStr::new(" ")));
}

#[test]
fn test_os_string_default() {
    let os_string: OsString = Default::default();
    assert_eq!("", &os_string);
}

#[test]
fn test_os_str_is_empty() {
    let mut os_string = OsString::new();
    assert!(os_string.is_empty());

    os_string.push("abc");
    assert!(!os_string.is_empty());

    os_string.clear();
    assert!(os_string.is_empty());
}

#[test]
fn test_os_str_len() {
    let mut os_string = OsString::new();
    assert_eq!(0, os_string.len());

    os_string.push("abc");
    assert_eq!(3, os_string.len());

    os_string.clear();
    assert_eq!(0, os_string.len());
}

#[test]
fn test_os_str_default() {
    let os_str: &OsStr = Default::default();
    assert_eq!("", os_str);
}

#[test]
fn into_boxed() {
    let orig = "Hello, world!";
    let os_str = OsStr::new(orig);
    let boxed: Box<OsStr> = Box::from(os_str);
    let os_string = os_str.to_owned().into_boxed_os_str().into_os_string();
    assert_eq!(os_str, &*boxed);
    assert_eq!(&*boxed, &*os_string);
    assert_eq!(&*os_string, os_str);
}

#[test]
fn boxed_default() {
    let boxed = <Box<OsStr>>::default();
    assert!(boxed.is_empty());
}

#[test]
fn test_os_str_clone_into() {
    let mut os_string = OsString::with_capacity(123);
    os_string.push("hello");
    let os_str = OsStr::new("bonjour");
    os_str.clone_into(&mut os_string);
    assert_eq!(os_str, os_string);
    assert!(os_string.capacity() >= 123);
}

#[test]
fn into_rc() {
    let orig = "Hello, world!";
    let os_str = OsStr::new(orig);
    let rc: Rc<OsStr> = Rc::from(os_str);
    let arc: Arc<OsStr> = Arc::from(os_str);

    assert_eq!(&*rc, os_str);
    assert_eq!(&*arc, os_str);

    let rc2: Rc<OsStr> = Rc::from(os_str.to_owned());
    let arc2: Arc<OsStr> = Arc::from(os_str.to_owned());

    assert_eq!(&*rc2, os_str);
    assert_eq!(&*arc2, os_str);
}

#[test]
fn slice_encoded_bytes() {
    let os_str = OsStr::new("123θგ🦀");
    // ASCII
    let digits = os_str.slice_encoded_bytes(..3);
    assert_eq!(digits, "123");
    let three = os_str.slice_encoded_bytes(2..3);
    assert_eq!(three, "3");
    // 2-byte UTF-8
    let theta = os_str.slice_encoded_bytes(3..5);
    assert_eq!(theta, "θ");
    // 3-byte UTF-8
    let gani = os_str.slice_encoded_bytes(5..8);
    assert_eq!(gani, "გ");
    // 4-byte UTF-8
    let crab = os_str.slice_encoded_bytes(8..);
    assert_eq!(crab, "🦀");
}

#[test]
#[should_panic]
fn slice_out_of_bounds() {
    let crab = OsStr::new("🦀");
    let _ = crab.slice_encoded_bytes(..5);
}

#[test]
#[should_panic]
fn slice_mid_char() {
    let crab = OsStr::new("🦀");
    let _ = crab.slice_encoded_bytes(..2);
}

#[cfg(unix)]
#[test]
#[should_panic(expected = "byte index 1 is not an OsStr boundary")]
fn slice_invalid_data() {
    use crate::os::unix::ffi::OsStrExt;

    let os_string = OsStr::from_bytes(b"\xFF\xFF");
    let _ = os_string.slice_encoded_bytes(1..);
}

#[cfg(unix)]
#[test]
#[should_panic(expected = "byte index 1 is not an OsStr boundary")]
fn slice_partial_utf8() {
    use crate::os::unix::ffi::{OsStrExt, OsStringExt};

    let part_crab = OsStr::from_bytes(&"🦀".as_bytes()[..3]);
    let mut os_string = OsString::from_vec(vec![0xFF]);
    os_string.push(part_crab);
    let _ = os_string.slice_encoded_bytes(1..);
}

#[cfg(unix)]
#[test]
fn slice_invalid_edge() {
    use crate::os::unix::ffi::{OsStrExt, OsStringExt};

    let os_string = OsStr::from_bytes(b"a\xFFa");
    assert_eq!(os_string.slice_encoded_bytes(..1), "a");
    assert_eq!(os_string.slice_encoded_bytes(1..), OsStr::from_bytes(b"\xFFa"));
    assert_eq!(os_string.slice_encoded_bytes(..2), OsStr::from_bytes(b"a\xFF"));
    assert_eq!(os_string.slice_encoded_bytes(2..), "a");

    let os_string = OsStr::from_bytes(&"abc🦀".as_bytes()[..6]);
    assert_eq!(os_string.slice_encoded_bytes(..3), "abc");
    assert_eq!(os_string.slice_encoded_bytes(3..), OsStr::from_bytes(b"\xF0\x9F\xA6"));

    let mut os_string = OsString::from_vec(vec![0xFF]);
    os_string.push("🦀");
    assert_eq!(os_string.slice_encoded_bytes(..1), OsStr::from_bytes(b"\xFF"));
    assert_eq!(os_string.slice_encoded_bytes(1..), "🦀");
}

#[cfg(windows)]
#[test]
#[should_panic(expected = "byte index 3 lies between surrogate codepoints")]
fn slice_between_surrogates() {
    use crate::os::windows::ffi::OsStringExt;

    let os_string = OsString::from_wide(&[0xD800, 0xD800]);
    assert_eq!(os_string.as_encoded_bytes(), &[0xED, 0xA0, 0x80, 0xED, 0xA0, 0x80]);
    let _ = os_string.slice_encoded_bytes(..3);
}

#[cfg(windows)]
#[test]
fn slice_surrogate_edge() {
    use crate::os::windows::ffi::OsStringExt;

    let surrogate = OsString::from_wide(&[0xD800]);
    let mut pre_crab = surrogate.clone();
    pre_crab.push("🦀");
    assert_eq!(pre_crab.slice_encoded_bytes(..3), surrogate);
    assert_eq!(pre_crab.slice_encoded_bytes(3..), "🦀");

    let mut post_crab = OsString::from("🦀");
    post_crab.push(&surrogate);
    assert_eq!(post_crab.slice_encoded_bytes(..4), "🦀");
    assert_eq!(post_crab.slice_encoded_bytes(4..), surrogate);
}

#[test]
fn clone_to_uninit() {
    let a = OsStr::new("hello.txt");

    let mut storage = vec![MaybeUninit::<u8>::uninit(); size_of_val::<OsStr>(a)];
    unsafe { a.clone_to_uninit(ptr::from_mut::<[_]>(storage.as_mut_slice()).cast()) };
    assert_eq!(a.as_encoded_bytes(), unsafe { MaybeUninit::slice_assume_init_ref(&storage) });

    let mut b: Box<OsStr> = OsStr::new("world.exe").into();
    assert_eq!(size_of_val::<OsStr>(a), size_of_val::<OsStr>(&b));
    assert_ne!(a, &*b);
    unsafe { a.clone_to_uninit(ptr::from_mut::<OsStr>(&mut b).cast()) };
    assert_eq!(a, &*b);
}
