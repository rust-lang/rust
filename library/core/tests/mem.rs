use core::mem::*;

#[cfg(panic = "unwind")]
use std::rc::Rc;

#[test]
fn size_of_basic() {
    assert_eq!(size_of::<u8>(), 1);
    assert_eq!(size_of::<u16>(), 2);
    assert_eq!(size_of::<u32>(), 4);
    assert_eq!(size_of::<u64>(), 8);
}

#[test]
#[cfg(target_pointer_width = "16")]
fn size_of_16() {
    assert_eq!(size_of::<usize>(), 2);
    assert_eq!(size_of::<*const usize>(), 2);
}

#[test]
#[cfg(target_pointer_width = "32")]
fn size_of_32() {
    assert_eq!(size_of::<usize>(), 4);
    assert_eq!(size_of::<*const usize>(), 4);
}

#[test]
#[cfg(target_pointer_width = "64")]
fn size_of_64() {
    assert_eq!(size_of::<usize>(), 8);
    assert_eq!(size_of::<*const usize>(), 8);
}

#[test]
fn size_of_val_basic() {
    assert_eq!(size_of_val(&1u8), 1);
    assert_eq!(size_of_val(&1u16), 2);
    assert_eq!(size_of_val(&1u32), 4);
    assert_eq!(size_of_val(&1u64), 8);
}

#[test]
fn align_of_basic() {
    assert_eq!(align_of::<u8>(), 1);
    assert_eq!(align_of::<u16>(), 2);
    assert_eq!(align_of::<u32>(), 4);
}

#[test]
#[cfg(target_pointer_width = "16")]
fn align_of_16() {
    assert_eq!(align_of::<usize>(), 2);
    assert_eq!(align_of::<*const usize>(), 2);
}

#[test]
#[cfg(target_pointer_width = "32")]
fn align_of_32() {
    assert_eq!(align_of::<usize>(), 4);
    assert_eq!(align_of::<*const usize>(), 4);
}

#[test]
#[cfg(target_pointer_width = "64")]
fn align_of_64() {
    assert_eq!(align_of::<usize>(), 8);
    assert_eq!(align_of::<*const usize>(), 8);
}

#[test]
fn align_of_val_basic() {
    assert_eq!(align_of_val(&1u8), 1);
    assert_eq!(align_of_val(&1u16), 2);
    assert_eq!(align_of_val(&1u32), 4);
}

#[test]
fn test_swap() {
    let mut x = 31337;
    let mut y = 42;
    swap(&mut x, &mut y);
    assert_eq!(x, 42);
    assert_eq!(y, 31337);
}

#[test]
fn test_replace() {
    let mut x = Some("test".to_string());
    let y = replace(&mut x, None);
    assert!(x.is_none());
    assert!(y.is_some());
}

#[test]
fn test_transmute_copy() {
    assert_eq!(1, unsafe { transmute_copy(&1) });
}

#[test]
#[allow(dead_code)]
fn test_discriminant_send_sync() {
    enum Regular {
        A,
        B(i32),
    }
    enum NotSendSync {
        A(*const i32),
    }

    fn is_send_sync<T: Send + Sync>() {}

    is_send_sync::<Discriminant<Regular>>();
    is_send_sync::<Discriminant<NotSendSync>>();
}

#[test]
fn assume_init_good() {
    const TRUE: bool = unsafe { MaybeUninit::<bool>::new(true).assume_init() };

    assert!(TRUE);
}

#[test]
fn uninit_array_assume_init() {
    let mut array: [MaybeUninit<i16>; 5] = MaybeUninit::uninit_array();
    array[0].write(3);
    array[1].write(1);
    array[2].write(4);
    array[3].write(1);
    array[4].write(5);

    let array = unsafe { MaybeUninit::array_assume_init(array) };

    assert_eq!(array, [3, 1, 4, 1, 5]);

    let [] = unsafe { MaybeUninit::<!>::array_assume_init([]) };
}

#[test]
fn uninit_write_slice() {
    let mut dst = [MaybeUninit::new(255); 64];
    let src = [0; 64];

    assert_eq!(MaybeUninit::write_slice(&mut dst, &src), &src);
}

#[test]
#[should_panic(expected = "source slice length (32) does not match destination slice length (64)")]
fn uninit_write_slice_panic_lt() {
    let mut dst = [MaybeUninit::uninit(); 64];
    let src = [0; 32];

    MaybeUninit::write_slice(&mut dst, &src);
}

#[test]
#[should_panic(expected = "source slice length (128) does not match destination slice length (64)")]
fn uninit_write_slice_panic_gt() {
    let mut dst = [MaybeUninit::uninit(); 64];
    let src = [0; 128];

    MaybeUninit::write_slice(&mut dst, &src);
}

#[test]
fn uninit_clone_from_slice() {
    let mut dst = [MaybeUninit::new(255); 64];
    let src = [0; 64];

    assert_eq!(MaybeUninit::write_slice_cloned(&mut dst, &src), &src);
}

#[test]
#[should_panic(expected = "destination and source slices have different lengths")]
fn uninit_write_slice_cloned_panic_lt() {
    let mut dst = [MaybeUninit::uninit(); 64];
    let src = [0; 32];

    MaybeUninit::write_slice_cloned(&mut dst, &src);
}

#[test]
#[should_panic(expected = "destination and source slices have different lengths")]
fn uninit_write_slice_cloned_panic_gt() {
    let mut dst = [MaybeUninit::uninit(); 64];
    let src = [0; 128];

    MaybeUninit::write_slice_cloned(&mut dst, &src);
}

#[test]
#[cfg(panic = "unwind")]
fn uninit_write_slice_cloned_mid_panic() {
    use std::panic;

    enum IncrementOrPanic {
        Increment(Rc<()>),
        ExpectedPanic,
        UnexpectedPanic,
    }

    impl Clone for IncrementOrPanic {
        fn clone(&self) -> Self {
            match self {
                Self::Increment(rc) => Self::Increment(rc.clone()),
                Self::ExpectedPanic => panic!("expected panic on clone"),
                Self::UnexpectedPanic => panic!("unexpected panic on clone"),
            }
        }
    }

    let rc = Rc::new(());

    let mut dst = [
        MaybeUninit::uninit(),
        MaybeUninit::uninit(),
        MaybeUninit::uninit(),
        MaybeUninit::uninit(),
    ];

    let src = [
        IncrementOrPanic::Increment(rc.clone()),
        IncrementOrPanic::Increment(rc.clone()),
        IncrementOrPanic::ExpectedPanic,
        IncrementOrPanic::UnexpectedPanic,
    ];

    let err = panic::catch_unwind(panic::AssertUnwindSafe(|| {
        MaybeUninit::write_slice_cloned(&mut dst, &src);
    }));

    drop(src);

    match err {
        Ok(_) => unreachable!(),
        Err(payload) => {
            payload
                .downcast::<&'static str>()
                .and_then(|s| if *s == "expected panic on clone" { Ok(s) } else { Err(s) })
                .unwrap_or_else(|p| panic::resume_unwind(p));

            assert_eq!(Rc::strong_count(&rc), 1)
        }
    }
}

#[test]
fn uninit_write_slice_cloned_no_drop() {
    #[derive(Clone)]
    struct Bomb;

    impl Drop for Bomb {
        fn drop(&mut self) {
            panic!("dropped a bomb! kaboom")
        }
    }

    let mut dst = [MaybeUninit::uninit()];
    let src = [Bomb];

    MaybeUninit::write_slice_cloned(&mut dst, &src);

    forget(src);
}

#[test]
fn uninit_const_assume_init_read() {
    const FOO: u32 = unsafe { MaybeUninit::new(42).assume_init_read() };
    assert_eq!(FOO, 42);
}

#[test]
fn const_maybe_uninit() {
    use std::ptr;

    #[derive(Debug, PartialEq)]
    struct Foo {
        x: u8,
        y: u8,
    }

    const FIELD_BY_FIELD: Foo = unsafe {
        let mut val = MaybeUninit::uninit();
        init_y(&mut val); // order shouldn't matter
        init_x(&mut val);
        val.assume_init()
    };

    const fn init_x(foo: &mut MaybeUninit<Foo>) {
        unsafe {
            *ptr::addr_of_mut!((*foo.as_mut_ptr()).x) = 1;
        }
    }

    const fn init_y(foo: &mut MaybeUninit<Foo>) {
        unsafe {
            *ptr::addr_of_mut!((*foo.as_mut_ptr()).y) = 2;
        }
    }

    assert_eq!(FIELD_BY_FIELD, Foo { x: 1, y: 2 });
}
