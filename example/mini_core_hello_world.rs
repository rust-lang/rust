// Adapted from https://github.com/sunfishcode/mir2cranelift/blob/master/rust-examples/nocore-hello-world.rs

#![feature(no_core, unboxed_closures, start, lang_items, box_syntax)]
#![no_core]
#![allow(dead_code)]

extern crate mini_core;

use mini_core::*;

#[link(name = "c")]
extern "C" {
    fn puts(s: *const u8);
}

unsafe extern "C" fn my_puts(s: *const u8) {
    puts(s);
}

// TODO remove when jit supports linking rlibs
#[cfg(jit)]
extern "C" fn panic<T>(_: T) -> ! {
    unsafe {
        intrinsics::abort();
    }
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

enum Ordering {
    Less = -1,
    Equal = 0,
    Greater = 1,
}

#[lang = "start"]
fn start<T: Termination + 'static>(
    main: fn() -> T,
    _argc: isize,
    _argv: *const *const u8,
) -> isize {
    main().report() as isize
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

fn main() {
    unsafe {
        let hello: &[u8] = b"Hello\0" as &[u8; 6];
        let ptr: *const u8 = hello as *const [u8] as *const u8;
        puts(ptr);

        // TODO remove when jit supports linking rlibs
        #[cfg(not(jit))]
        {
            let world: Box<&str> = box "World!\0";
            puts(*world as *const str as *const u8);
            world as Box<SomeTrait>;
        }

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
    }

    let _ = NoisyDrop {
        text: "Outer got dropped!\0",
        inner: NoisyDropInner,
    };

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
}
