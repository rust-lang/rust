use core::clone::CloneToUninit;
use core::ffi::CStr;
use core::mem::MaybeUninit;
use core::ptr;

#[test]
#[allow(suspicious_double_ref_op)]
fn test_borrowed_clone() {
    let x = 5;
    let y: &i32 = &x;
    let z: &i32 = (&y).clone();
    assert_eq!(*z, 5);
}

#[test]
fn test_clone_from() {
    let a = Box::new(5);
    let mut b = Box::new(10);
    b.clone_from(&a);
    assert_eq!(*b, 5);
}

#[test]
fn test_clone_to_uninit_slice_success() {
    // Using `String`s to exercise allocation and Drop of the individual elements;
    // if something is aliased or double-freed, at least Miri will catch that.
    let a: [String; 3] = ["a", "b", "c"].map(String::from);

    let mut storage: MaybeUninit<[String; 3]> = MaybeUninit::uninit();
    let b: [String; 3] = unsafe {
        a[..].clone_to_uninit(storage.as_mut_ptr().cast());
        storage.assume_init()
    };

    assert_eq!(a, b);
}

#[test]
#[cfg(panic = "unwind")]
fn test_clone_to_uninit_slice_drops_on_panic() {
    use core::sync::atomic::AtomicUsize;
    use core::sync::atomic::Ordering::Relaxed;

    /// A static counter is OK to use as long as _this one test_ isn't run several times in
    /// multiple threads.
    static COUNTER: AtomicUsize = AtomicUsize::new(0);
    /// Counts how many instances are live, and panics if a fifth one is created
    struct CountsDropsAndPanics {}
    impl CountsDropsAndPanics {
        fn new() -> Self {
            COUNTER.fetch_add(1, Relaxed);
            Self {}
        }
    }
    impl Clone for CountsDropsAndPanics {
        fn clone(&self) -> Self {
            if COUNTER.load(Relaxed) == 4 { panic!("intentional panic") } else { Self::new() }
        }
    }
    impl Drop for CountsDropsAndPanics {
        fn drop(&mut self) {
            COUNTER.fetch_sub(1, Relaxed);
        }
    }

    let a: [CountsDropsAndPanics; 3] = core::array::from_fn(|_| CountsDropsAndPanics::new());
    assert_eq!(COUNTER.load(Relaxed), 3);

    let panic_payload = std::panic::catch_unwind(|| {
        let mut storage: MaybeUninit<[CountsDropsAndPanics; 3]> = MaybeUninit::uninit();
        // This should panic halfway through
        unsafe {
            a[..].clone_to_uninit(storage.as_mut_ptr().cast());
        }
    })
    .unwrap_err();
    assert_eq!(panic_payload.downcast().unwrap(), Box::new("intentional panic"));

    // Check for lack of leak, which is what this test is looking for
    assert_eq!(COUNTER.load(Relaxed), 3, "leaked during clone!");

    // Might as well exercise the rest of the drops
    drop(a);
    assert_eq!(COUNTER.load(Relaxed), 0);
}

#[test]
fn test_clone_to_uninit_str() {
    let a = "hello";

    let mut storage: MaybeUninit<[u8; 5]> = MaybeUninit::uninit();
    unsafe { a.clone_to_uninit(storage.as_mut_ptr().cast()) };
    assert_eq!(a.as_bytes(), unsafe { storage.assume_init() }.as_slice());

    let mut b: Box<str> = "world".into();
    assert_eq!(a.len(), b.len());
    assert_ne!(a, &*b);
    unsafe { a.clone_to_uninit(ptr::from_mut::<str>(&mut b).cast()) };
    assert_eq!(a, &*b);
}

#[test]
fn test_clone_to_uninit_cstr() {
    let a = c"hello";

    let mut storage: MaybeUninit<[u8; 6]> = MaybeUninit::uninit();
    unsafe { a.clone_to_uninit(storage.as_mut_ptr().cast()) };
    assert_eq!(a.to_bytes_with_nul(), unsafe { storage.assume_init() }.as_slice());

    let mut b: Box<CStr> = c"world".into();
    assert_eq!(a.count_bytes(), b.count_bytes());
    assert_ne!(a, &*b);
    unsafe { a.clone_to_uninit(ptr::from_mut::<CStr>(&mut b).cast()) };
    assert_eq!(a, &*b);
}

#[test]
fn cstr_metadata_is_length_with_nul() {
    let s: &CStr = c"abcdef";
    let p: *const CStr = ptr::from_ref(s);
    let bytes: *const [u8] = p as *const [u8];
    assert_eq!(s.to_bytes_with_nul().len(), bytes.len());
}

#[test]
fn test_const_clone() {
    const {
        let bool: bool = Default::default();
        let char: char = Default::default();
        let ascii_char: std::ascii::Char = Default::default();
        let usize: usize = Default::default();
        let u8: u8 = Default::default();
        let u16: u16 = Default::default();
        let u32: u32 = Default::default();
        let u64: u64 = Default::default();
        let u128: u128 = Default::default();
        let i8: i8 = Default::default();
        let i16: i16 = Default::default();
        let i32: i32 = Default::default();
        let i64: i64 = Default::default();
        let i128: i128 = Default::default();
        let f16: f16 = Default::default();
        let f32: f32 = Default::default();
        let f64: f64 = Default::default();
        let f128: f128 = Default::default();

        let bool_clone: bool = bool.clone();
        let char_clone: char = char.clone();
        let ascii_char_clone: std::ascii::Char = ascii_char.clone();

        let usize_clone: usize = usize.clone();
        let u8_clone: u8 = u8.clone();
        let u16_clone: u16 = u16.clone();
        let u32_clone: u32 = u32.clone();
        let u64_clone: u64 = u64.clone();
        let u128_clone: u128 = u128.clone();
        let i8_clone: i8 = i8.clone();
        let i16_clone: i16 = i16.clone();
        let i32_clone: i32 = i32.clone();
        let i64_clone: i64 = i64.clone();
        let i128_clone: i128 = i128.clone();
        let f16_clone: f16 = f16.clone();
        let f32_clone: f32 = f32.clone();
        let f64_clone: f64 = f64.clone();
        let f128_clone: f128 = f128.clone();

        assert!(bool == bool_clone);
        assert!(char == char_clone);
        assert!(ascii_char == ascii_char_clone);
        assert!(usize == usize_clone);
        assert!(u8 == u8_clone);
        assert!(u16 == u16_clone);
        assert!(u32 == u32_clone);
        assert!(u64 == u64_clone);
        assert!(u128 == u128_clone);
        assert!(i8 == i8_clone);
        assert!(i16 == i16_clone);
        assert!(i32 == i32_clone);
        assert!(i64 == i64_clone);
        assert!(i128 == i128_clone);
        assert!(f16 == f16_clone);
        assert!(f32 == f32_clone);
        assert!(f64 == f64_clone);
        assert!(f128 == f128_clone);

        let src: [i32; 4] = [1, 2, 3, 4];
        let mut dst: [i32; 2] = [0, 0];

        dst.clone_from_slice(&src[2..]);

        assert!(src == [1, 2, 3, 4]);
        assert!(dst == [3, 4]);
    }
}
