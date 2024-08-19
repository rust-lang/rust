use core::ffi::os_str::OsStr;
use core::mem::MaybeUninit;
use core::ptr;

#[test]
fn test_os_str_default() {
    let os_str: &OsStr = Default::default();
    assert_eq!("", os_str);
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
fn clone_to_uninit() {
    let a = OsStr::new("hello.txt");

    let mut storage = vec![MaybeUninit::<u8>::uninit(); size_of_val::<OsStr>(a)];
    unsafe { a.clone_to_uninit(ptr::from_mut::<[_]>(storage.as_mut_slice()) as *mut OsStr) };
    assert_eq!(a.as_encoded_bytes(), unsafe { MaybeUninit::slice_assume_init_ref(&storage) });

    let mut b: Box<OsStr> = OsStr::new("world.exe").into();
    assert_eq!(size_of_val::<OsStr>(a), size_of_val::<OsStr>(&b));
    assert_ne!(a, &*b);
    unsafe { a.clone_to_uninit(ptr::from_mut::<OsStr>(&mut b)) };
    assert_eq!(a, &*b);
}
