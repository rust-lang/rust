// Adapted from https://github.com/sunfishcode/mir2cranelift/blob/master/rust-examples/nocore-hello-world.rs

#![feature(no_core, unboxed_closures, start, lang_items, box_syntax, slice_patterns, never_type, linkage)]
#![no_core]
#![allow(dead_code)]

extern crate mini_core;

use mini_core::*;
use mini_core::libc::*;

unsafe extern "C" fn my_puts(s: *const u8) {
    puts(s);
}

#[lang = "termination"]
trait Termination {
    fn report(self) -> i32;
}

impl Termination for () {
    fn report(self) -> i32 {
        unsafe {
            NUM = 6 * 7 + 1 + (1u8 == 1u8) as u8; // 44
            *NUM_REF as i32
        }
    }
}

trait SomeTrait {
    fn object_safe(&self);
}

impl SomeTrait for &'static str {
    fn object_safe(&self) {
        unsafe {
            puts(*self as *const str as *const u8);
        }
    }
}

struct NoisyDrop {
    text: &'static str,
    inner: NoisyDropInner,
}

struct NoisyDropInner;

impl Drop for NoisyDrop {
    fn drop(&mut self) {
        unsafe {
            puts(self.text as *const str as *const u8);
        }
    }
}

impl Drop for NoisyDropInner {
    fn drop(&mut self) {
        unsafe {
            puts("Inner got dropped!\0" as *const str as *const u8);
        }
    }
}

impl SomeTrait for NoisyDrop {
    fn object_safe(&self) {}
}

enum Ordering {
    Less = -1,
    Equal = 0,
    Greater = 1,
}

#[lang = "start"]
fn start<T: Termination + 'static>(
    main: fn() -> T,
    argc: isize,
    argv: *const *const u8,
) -> isize {
    if argc == 3 {
        unsafe { puts(*argv); }
        unsafe { puts(*((argv as usize + intrinsics::size_of::<*const u8>()) as *const *const u8)); }
        unsafe { puts(*((argv as usize + 2 * intrinsics::size_of::<*const u8>()) as *const *const u8)); }
    }

    main().report();
    0
}

static mut NUM: u8 = 6 * 7;
static NUM_REF: &'static u8 = unsafe { &NUM };

macro_rules! assert {
    ($e:expr) => {
        if !$e {
            panic(&(stringify!(! $e), file!(), line!(), 0));
        }
    };
}

macro_rules! assert_eq {
    ($l:expr, $r: expr) => {
        if $l != $r {
            panic(&(stringify!($l != $r), file!(), line!(), 0));
        }
    }
}

struct Unique<T: ?Sized> {
    pointer: *const T,
    _marker: PhantomData<T>,
}

impl<T: ?Sized, U: ?Sized> CoerceUnsized<Unique<U>> for Unique<T> where T: Unsize<U> {}

fn take_f32(_f: f32) {}
fn take_unique(_u: Unique<()>) {}

fn main() {
    take_unique(Unique {
        pointer: 0 as *const (),
        _marker: PhantomData,
    });
    take_f32(0.1);

    //return;

    unsafe {
        printf("Hello %s\n\0" as *const str as *const i8, "printf\0" as *const str as *const i8);

        let hello: &[u8] = b"Hello\0" as &[u8; 6];
        let ptr: *const u8 = hello as *const [u8] as *const u8;
        puts(ptr);

        let world: Box<&str> = box "World!\0";
        puts(*world as *const str as *const u8);
        world as Box<SomeTrait>;

        assert_eq!(intrinsics::bitreverse(0b10101000u8), 0b00010101u8);

        assert_eq!(intrinsics::bswap(0xabu8), 0xabu8);
        assert_eq!(intrinsics::bswap(0xddccu16), 0xccddu16);
        assert_eq!(intrinsics::bswap(0xffee_ddccu32), 0xccdd_eeffu32);
        assert_eq!(intrinsics::bswap(0x1234_5678_ffee_ddccu64), 0xccdd_eeff_7856_3412u64);

        assert_eq!(intrinsics::size_of_val(hello) as u8, 6);

        let chars = &['C', 'h', 'a', 'r', 's'];
        let chars = chars as &[char];
        assert_eq!(intrinsics::size_of_val(chars) as u8, 4 * 5);

        let a: &dyn SomeTrait = &"abc\0";
        a.object_safe();

        assert_eq!(intrinsics::size_of_val(a) as u8, 16);
        assert_eq!(intrinsics::size_of_val(&0u32) as u8, 4);

        assert_eq!(intrinsics::min_align_of::<u16>() as u8, 2);
        assert_eq!(intrinsics::min_align_of_val(&a) as u8, intrinsics::min_align_of::<&str>() as u8);

        assert!(!intrinsics::needs_drop::<u8>());
        assert!(intrinsics::needs_drop::<NoisyDrop>());

        Unique {
            pointer: 0 as *const &str,
            _marker: PhantomData,
        } as Unique<dyn SomeTrait>;

        struct MyDst<T: ?Sized>(T);

        intrinsics::size_of_val(&MyDst([0u8; 4]) as &MyDst<[u8]>);

        struct Foo {
            x: u8,
            y: !,
        }

        unsafe fn zeroed<T>() -> T {
            intrinsics::init::<T>()
        }

        unsafe fn uninitialized<T>() -> T {
            intrinsics::uninit::<T>()
        }

        #[allow(unreachable_code)]
        {
            if false {
                zeroed::<!>();
                zeroed::<Foo>();
                zeroed::<(u8, u8)>();
                uninitialized::<Foo>();
            }
        }
    }

    let _ = box NoisyDrop {
        text: "Boxed outer got dropped!\0",
        inner: NoisyDropInner,
    } as Box<SomeTrait>;

    const FUNC_REF: Option<fn()> = Some(main);
    match FUNC_REF {
        Some(_) => {},
        None => assert!(false),
    }

    match Ordering::Less {
        Ordering::Less => {},
        _ => assert!(false),
    }

    [NoisyDropInner, NoisyDropInner];

    let x = &[0u32, 42u32] as &[u32];
    match x {
        [] => assert_eq!(0u32, 1),
        [_, ref y..] => assert_eq!(&x[1] as *const u32 as usize, &y[0] as *const u32 as usize),
    }

    assert_eq!(((|()| 42u8) as fn(()) -> u8)(()), 42);

    extern {
        #[linkage = "weak"]
        static ABC: *const u8;
    }

    {
        extern {
            #[linkage = "weak"]
            static ABC: *const u8;
        }
    }

    unsafe { assert_eq!(ABC as usize, 0); }

    &mut (|| Some(0 as *const ())) as &mut FnMut() -> Option<*const ()>;
}
