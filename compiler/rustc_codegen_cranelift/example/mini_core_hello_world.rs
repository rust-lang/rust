#![feature(
    no_core,
    lang_items,
    never_type,
    linkage,
    extern_types,
    naked_functions,
    thread_local,
    repr_simd
)]
#![no_core]
#![allow(dead_code, non_camel_case_types, internal_features)]

extern crate mini_core;

use mini_core::libc::*;
use mini_core::*;

macro_rules! assert {
    ($e:expr) => {
        if !$e {
            panic(stringify!(!$e));
        }
    };
}

macro_rules! assert_eq {
    ($l:expr, $r: expr) => {
        if $l != $r {
            panic(stringify!($l != $r));
        }
    };
}

#[lang = "termination"]
trait Termination {
    fn report(self) -> i32;
}

impl Termination for () {
    fn report(self) -> i32 {
        unsafe {
            NUM = 6 * 7 + 1 + (1u8 == 1u8) as u8; // 44
            assert_eq!(*NUM_REF as i32, 44);
        }
        0
    }
}

trait SomeTrait {
    fn object_safe(&self);
}

impl SomeTrait for &'static str {
    fn object_safe(&self) {
        unsafe {
            puts(*self as *const str as *const i8);
        }
    }
}

struct NoisyDrop {
    text: &'static str,
    inner: NoisyDropInner,
}

struct NoisyDropUnsized {
    inner: NoisyDropInner,
    text: str,
}

struct NoisyDropInner;

impl Drop for NoisyDrop {
    fn drop(&mut self) {
        unsafe {
            puts(self.text as *const str as *const i8);
        }
    }
}

impl Drop for NoisyDropInner {
    fn drop(&mut self) {
        unsafe {
            puts("Inner got dropped!\0" as *const str as *const i8);
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
    _sigpipe: u8,
) -> isize {
    if argc == 3 {
        unsafe {
            puts(*argv as *const i8);
        }
        unsafe {
            puts(*((argv as usize + intrinsics::size_of::<*const u8>()) as *const *const i8));
        }
        unsafe {
            puts(*((argv as usize + 2 * intrinsics::size_of::<*const u8>()) as *const *const i8));
        }
    }

    main().report() as isize
}

static mut NUM: u8 = 6 * 7;

static NUM_REF: &'static u8 = unsafe { &*&raw const NUM };

unsafe fn zeroed<T>() -> T {
    let mut uninit = MaybeUninit { uninit: () };
    intrinsics::write_bytes(&mut uninit.value.value as *mut T, 0, 1);
    uninit.value.value
}

fn take_f32(_f: f32) {}
fn take_unique(_u: Unique<()>) {}

fn return_u128_pair() -> (u128, u128) {
    (0, 0)
}

fn call_return_u128_pair() {
    return_u128_pair();
}

#[repr(C)]
pub struct bool_11 {
    field0: bool,
    field1: bool,
    field2: bool,
    field3: bool,
    field4: bool,
    field5: bool,
    field6: bool,
    field7: bool,
    field8: bool,
    field9: bool,
    field10: bool,
}

extern "C" fn bool_struct_in_11(_arg0: bool_11) {}

#[allow(unreachable_code)] // FIXME false positive
fn main() {
    take_unique(Unique { pointer: unsafe { NonNull(1 as *mut ()) }, _marker: PhantomData });
    take_f32(0.1);

    call_return_u128_pair();

    bool_struct_in_11(bool_11 {
        field0: true,
        field1: true,
        field2: true,
        field3: true,
        field4: true,
        field5: true,
        field6: true,
        field7: true,
        field8: true,
        field9: true,
        field10: true,
    });

    let slice = &[0, 1] as &[i32];
    let slice_ptr = slice as *const [i32] as *const i32;

    assert_eq!(slice_ptr as usize % 4, 0);

    unsafe {
        printf("Hello %s\n\0" as *const str as *const i8, "printf\0" as *const str as *const i8);

        let hello: &[u8] = b"Hello\0" as &[u8; 6];
        let ptr: *const i8 = hello as *const [u8] as *const i8;
        puts(ptr);

        let world: Box<&str> = Box::new("World!\0");
        puts(*world as *const str as *const i8);
        world as Box<dyn SomeTrait>;

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
        assert_eq!(
            intrinsics::min_align_of_val(&a) as u8,
            intrinsics::min_align_of::<&str>() as u8
        );

        assert!(!intrinsics::needs_drop::<u8>());
        assert!(!intrinsics::needs_drop::<[u8]>());
        assert!(intrinsics::needs_drop::<NoisyDrop>());
        assert!(intrinsics::needs_drop::<NoisyDropUnsized>());

        Unique { pointer: NonNull(1 as *mut &str), _marker: PhantomData } as Unique<dyn SomeTrait>;

        struct MyDst<T: ?Sized>(T);

        intrinsics::size_of_val(&MyDst([0u8; 4]) as &MyDst<[u8]>);

        struct Foo {
            x: u8,
            y: !,
        }

        unsafe fn uninitialized<T>() -> T {
            MaybeUninit { uninit: () }.value.value
        }

        zeroed::<(u8, u8)>();
        #[allow(unreachable_code)]
        {
            if false {
                zeroed::<!>();
                zeroed::<Foo>();
                uninitialized::<Foo>();
            }
        }
    }

    let _ = Box::new(NoisyDrop { text: "Boxed outer got dropped!\0", inner: NoisyDropInner })
        as Box<dyn SomeTrait>;

    const FUNC_REF: Option<fn()> = Some(main);
    match FUNC_REF {
        Some(_) => {}
        None => assert!(false),
    }

    match Ordering::Less {
        Ordering::Less => {}
        _ => assert!(false),
    }

    [NoisyDropInner, NoisyDropInner];

    let x = &[0u32, 42u32] as &[u32];
    match x {
        [] => assert_eq!(0u32, 1),
        [_, ref y @ ..] => assert_eq!(&x[1] as *const u32 as usize, &y[0] as *const u32 as usize),
    }

    assert_eq!(((|()| 42u8) as fn(()) -> u8)(()), 42);

    #[cfg(not(any(jit, windows)))]
    {
        extern "C" {
            #[linkage = "extern_weak"]
            static ABC: *const u8;
        }

        {
            extern "C" {
                #[linkage = "extern_weak"]
                static ABC: *const u8;
            }
        }

        unsafe {
            assert_eq!(ABC as usize, 0);
        }
    }

    &mut (|| Some(0 as *const ())) as &mut dyn FnMut() -> Option<*const ()>;

    let f = 1000.0;
    assert_eq!(f as u8, 255);
    let f2 = -1000.0;
    assert_eq!(f2 as i8, -128);
    assert_eq!(f2 as u8, 0);

    let amount = 0;
    assert_eq!(1u128 << amount, 1);

    static ANOTHER_STATIC: &u8 = &A_STATIC;
    assert_eq!(*ANOTHER_STATIC, 42);

    check_niche_behavior();

    extern "C" {
        type ExternType;
    }

    struct ExternTypeWrapper {
        _a: ExternType,
    }

    let nullptr = 0 as *const ();
    let extern_nullptr = nullptr as *const ExternTypeWrapper;
    extern_nullptr as *const ();
    let slice_ptr = &[] as *const [u8];
    slice_ptr as *const u8;

    let repeat = [Some(42); 2];
    assert_eq!(repeat[0], Some(42));
    assert_eq!(repeat[1], Some(42));

    from_decimal_string();

    #[cfg(all(not(jit), not(all(windows, target_env = "gnu"))))]
    test_tls();

    #[cfg(all(not(jit), target_arch = "x86_64", any(target_os = "linux", target_os = "macos")))]
    unsafe {
        global_asm_test();
        naked_test();
    }

    // Both statics have a reference that points to the same anonymous allocation.
    static REF1: &u8 = &42;
    static REF2: &u8 = REF1;
    assert_eq!(*REF1, *REF2);

    #[repr(simd)]
    struct V([f64; 2]);

    let f = V([0.0, 1.0]);
    let _a = f.0[0];

    stack_val_align();
}

#[inline(never)]
fn stack_val_align() {
    #[repr(align(8192))]
    struct Foo(u8);

    let a = Foo(0);
    assert_eq!(&a as *const Foo as usize % 8192, 0);
}

#[cfg(all(not(jit), target_arch = "x86_64", any(target_os = "linux", target_os = "macos")))]
extern "C" {
    fn global_asm_test();
}

#[cfg(all(not(jit), target_arch = "x86_64", target_os = "linux"))]
global_asm! {
    "
    .global global_asm_test
    global_asm_test:
    // comment that would normally be removed by LLVM
    ret
    "
}

#[cfg(all(not(jit), target_arch = "x86_64", target_os = "macos"))]
global_asm! {
    "
    .global _global_asm_test
    _global_asm_test:
    // comment that would normally be removed by LLVM
    ret
    "
}

#[cfg(all(not(jit), target_arch = "x86_64"))]
#[naked]
extern "C" fn naked_test() {
    unsafe {
        naked_asm!("ret");
    }
}

#[repr(C)]
enum c_void {
    _1,
    _2,
}

type c_int = i32;
type c_ulong = u64;

type pthread_t = c_ulong;

#[repr(C)]
struct pthread_attr_t {
    __size: [u64; 7],
}

#[link(name = "pthread")]
#[cfg(unix)]
extern "C" {
    fn pthread_attr_init(attr: *mut pthread_attr_t) -> c_int;

    fn pthread_create(
        native: *mut pthread_t,
        attr: *const pthread_attr_t,
        f: extern "C" fn(_: *mut c_void) -> *mut c_void,
        value: *mut c_void,
    ) -> c_int;

    fn pthread_join(native: pthread_t, value: *mut *mut c_void) -> c_int;
}

type DWORD = u32;
type LPDWORD = *mut u32;

type LPVOID = *mut c_void;
type HANDLE = *mut c_void;

#[link(name = "msvcrt")]
#[cfg(windows)]
extern "C" {
    fn WaitForSingleObject(hHandle: LPVOID, dwMilliseconds: DWORD) -> DWORD;

    fn CreateThread(
        lpThreadAttributes: LPVOID, // Technically LPSECURITY_ATTRIBUTES, but we don't use it anyway
        dwStackSize: usize,
        lpStartAddress: extern "C" fn(_: *mut c_void) -> *mut c_void,
        lpParameter: LPVOID,
        dwCreationFlags: DWORD,
        lpThreadId: LPDWORD,
    ) -> HANDLE;
}

struct Thread {
    #[cfg(windows)]
    handle: HANDLE,
    #[cfg(unix)]
    handle: pthread_t,
}

impl Thread {
    unsafe fn create(f: extern "C" fn(_: *mut c_void) -> *mut c_void) -> Self {
        #[cfg(unix)]
        {
            let mut attr: pthread_attr_t = zeroed();
            let mut thread: pthread_t = 0;

            if pthread_attr_init(&mut attr) != 0 {
                assert!(false);
            }

            if pthread_create(&mut thread, &attr, f, 0 as *mut c_void) != 0 {
                assert!(false);
            }

            Thread { handle: thread }
        }

        #[cfg(windows)]
        {
            let handle = CreateThread(0 as *mut c_void, 0, f, 0 as *mut c_void, 0, 0 as *mut u32);

            if (handle as u64) == 0 {
                assert!(false);
            }

            Thread { handle }
        }
    }

    unsafe fn join(self) {
        #[cfg(unix)]
        {
            let mut res = 0 as *mut c_void;
            pthread_join(self.handle, &mut res);
        }

        #[cfg(windows)]
        {
            // The INFINITE macro is used to signal operations that do not timeout.
            let infinite = 0xffffffff;
            assert!(WaitForSingleObject(self.handle, infinite) == 0);
        }
    }
}

#[thread_local]
#[cfg(not(jit))]
static mut TLS: u8 = 42;

#[cfg(not(jit))]
extern "C" fn mutate_tls(_: *mut c_void) -> *mut c_void {
    unsafe {
        TLS = 0;
    }
    0 as *mut c_void
}

#[cfg(not(jit))]
fn test_tls() {
    unsafe {
        assert_eq!(TLS, 42);

        let thread = Thread::create(mutate_tls);
        thread.join();

        // TLS of main thread must not have been changed by the other thread.
        assert_eq!(TLS, 42);

        puts("TLS works!\n\0" as *const str as *const i8);
    }
}

// Copied ui/issues/issue-61696.rs

pub enum Infallible {}

// The check that the `bool` field of `V1` is encoding a "niche variant"
// (i.e. not `V1`, so `V3` or `V4`) used to be mathematically incorrect,
// causing valid `V1` values to be interpreted as other variants.
pub enum E1 {
    V1 { f: bool },
    V2 { f: Infallible },
    V3,
    V4,
}

// Computing the discriminant used to be done using the niche type (here `u8`,
// from the `bool` field of `V1`), overflowing for variants with large enough
// indices (`V3` and `V4`), causing them to be interpreted as other variants.
#[rustfmt::skip]
pub enum E2<X> {
    V1 { f: bool },

    /*_00*/ _01(X), _02(X), _03(X), _04(X), _05(X), _06(X), _07(X),
    _08(X), _09(X), _0A(X), _0B(X), _0C(X), _0D(X), _0E(X), _0F(X),
    _10(X), _11(X), _12(X), _13(X), _14(X), _15(X), _16(X), _17(X),
    _18(X), _19(X), _1A(X), _1B(X), _1C(X), _1D(X), _1E(X), _1F(X),
    _20(X), _21(X), _22(X), _23(X), _24(X), _25(X), _26(X), _27(X),
    _28(X), _29(X), _2A(X), _2B(X), _2C(X), _2D(X), _2E(X), _2F(X),
    _30(X), _31(X), _32(X), _33(X), _34(X), _35(X), _36(X), _37(X),
    _38(X), _39(X), _3A(X), _3B(X), _3C(X), _3D(X), _3E(X), _3F(X),
    _40(X), _41(X), _42(X), _43(X), _44(X), _45(X), _46(X), _47(X),
    _48(X), _49(X), _4A(X), _4B(X), _4C(X), _4D(X), _4E(X), _4F(X),
    _50(X), _51(X), _52(X), _53(X), _54(X), _55(X), _56(X), _57(X),
    _58(X), _59(X), _5A(X), _5B(X), _5C(X), _5D(X), _5E(X), _5F(X),
    _60(X), _61(X), _62(X), _63(X), _64(X), _65(X), _66(X), _67(X),
    _68(X), _69(X), _6A(X), _6B(X), _6C(X), _6D(X), _6E(X), _6F(X),
    _70(X), _71(X), _72(X), _73(X), _74(X), _75(X), _76(X), _77(X),
    _78(X), _79(X), _7A(X), _7B(X), _7C(X), _7D(X), _7E(X), _7F(X),
    _80(X), _81(X), _82(X), _83(X), _84(X), _85(X), _86(X), _87(X),
    _88(X), _89(X), _8A(X), _8B(X), _8C(X), _8D(X), _8E(X), _8F(X),
    _90(X), _91(X), _92(X), _93(X), _94(X), _95(X), _96(X), _97(X),
    _98(X), _99(X), _9A(X), _9B(X), _9C(X), _9D(X), _9E(X), _9F(X),
    _A0(X), _A1(X), _A2(X), _A3(X), _A4(X), _A5(X), _A6(X), _A7(X),
    _A8(X), _A9(X), _AA(X), _AB(X), _AC(X), _AD(X), _AE(X), _AF(X),
    _B0(X), _B1(X), _B2(X), _B3(X), _B4(X), _B5(X), _B6(X), _B7(X),
    _B8(X), _B9(X), _BA(X), _BB(X), _BC(X), _BD(X), _BE(X), _BF(X),
    _C0(X), _C1(X), _C2(X), _C3(X), _C4(X), _C5(X), _C6(X), _C7(X),
    _C8(X), _C9(X), _CA(X), _CB(X), _CC(X), _CD(X), _CE(X), _CF(X),
    _D0(X), _D1(X), _D2(X), _D3(X), _D4(X), _D5(X), _D6(X), _D7(X),
    _D8(X), _D9(X), _DA(X), _DB(X), _DC(X), _DD(X), _DE(X), _DF(X),
    _E0(X), _E1(X), _E2(X), _E3(X), _E4(X), _E5(X), _E6(X), _E7(X),
    _E8(X), _E9(X), _EA(X), _EB(X), _EC(X), _ED(X), _EE(X), _EF(X),
    _F0(X), _F1(X), _F2(X), _F3(X), _F4(X), _F5(X), _F6(X), _F7(X),
    _F8(X), _F9(X), _FA(X), _FB(X), _FC(X), _FD(X), _FE(X), _FF(X),

    V3,
    V4,
}

#[allow(unreachable_patterns)]
fn check_niche_behavior() {
    if let E1::V2 { .. } = (E1::V1 { f: true }) {
        intrinsics::abort();
    }

    if let E2::V1 { .. } = E2::V3::<Infallible> {
        intrinsics::abort();
    }
}

fn from_decimal_string() {
    loop {
        let multiplier = 1;

        take_multiplier_ref(&multiplier);

        if multiplier == 1 {
            break;
        }

        unreachable();
    }
}

fn take_multiplier_ref(_multiplier: &u128) {}

fn unreachable() -> ! {
    panic("unreachable")
}
