use core::sync::atomic::Ordering::SeqCst;
use core::sync::atomic::*;

#[test]
fn bool_() {
    let a = AtomicBool::new(false);
    assert_eq!(a.compare_exchange(false, true, SeqCst, SeqCst), Ok(false));
    assert_eq!(a.compare_exchange(false, true, SeqCst, SeqCst), Err(true));

    a.store(false, SeqCst);
    assert_eq!(a.compare_exchange(false, true, SeqCst, SeqCst), Ok(false));
}

#[test]
fn bool_and() {
    let a = AtomicBool::new(true);
    assert_eq!(a.fetch_and(false, SeqCst), true);
    assert_eq!(a.load(SeqCst), false);
}

#[test]
fn bool_nand() {
    let a = AtomicBool::new(false);
    assert_eq!(a.fetch_nand(false, SeqCst), false);
    assert_eq!(a.load(SeqCst), true);
    assert_eq!(a.fetch_nand(false, SeqCst), true);
    assert_eq!(a.load(SeqCst), true);
    assert_eq!(a.fetch_nand(true, SeqCst), true);
    assert_eq!(a.load(SeqCst), false);
    assert_eq!(a.fetch_nand(true, SeqCst), false);
    assert_eq!(a.load(SeqCst), true);
}

#[test]
fn uint_and() {
    let x = AtomicUsize::new(0xf731);
    assert_eq!(x.fetch_and(0x137f, SeqCst), 0xf731);
    assert_eq!(x.load(SeqCst), 0xf731 & 0x137f);
}

#[test]
fn uint_nand() {
    let x = AtomicUsize::new(0xf731);
    assert_eq!(x.fetch_nand(0x137f, SeqCst), 0xf731);
    assert_eq!(x.load(SeqCst), !(0xf731 & 0x137f));
}

#[test]
fn uint_or() {
    let x = AtomicUsize::new(0xf731);
    assert_eq!(x.fetch_or(0x137f, SeqCst), 0xf731);
    assert_eq!(x.load(SeqCst), 0xf731 | 0x137f);
}

#[test]
fn uint_xor() {
    let x = AtomicUsize::new(0xf731);
    assert_eq!(x.fetch_xor(0x137f, SeqCst), 0xf731);
    assert_eq!(x.load(SeqCst), 0xf731 ^ 0x137f);
}

#[test]
fn int_and() {
    let x = AtomicIsize::new(0xf731);
    assert_eq!(x.fetch_and(0x137f, SeqCst), 0xf731);
    assert_eq!(x.load(SeqCst), 0xf731 & 0x137f);
}

#[test]
fn int_nand() {
    let x = AtomicIsize::new(0xf731);
    assert_eq!(x.fetch_nand(0x137f, SeqCst), 0xf731);
    assert_eq!(x.load(SeqCst), !(0xf731 & 0x137f));
}

#[test]
fn int_or() {
    let x = AtomicIsize::new(0xf731);
    assert_eq!(x.fetch_or(0x137f, SeqCst), 0xf731);
    assert_eq!(x.load(SeqCst), 0xf731 | 0x137f);
}

#[test]
fn int_xor() {
    let x = AtomicIsize::new(0xf731);
    assert_eq!(x.fetch_xor(0x137f, SeqCst), 0xf731);
    assert_eq!(x.load(SeqCst), 0xf731 ^ 0x137f);
}

static S_FALSE: AtomicBool = AtomicBool::new(false);
static S_TRUE: AtomicBool = AtomicBool::new(true);
static S_INT: AtomicIsize = AtomicIsize::new(0);
static S_UINT: AtomicUsize = AtomicUsize::new(0);

#[test]
fn static_init() {
    // Note that we're not really testing the mutability here but it's important
    // on Android at the moment (#49775)
    assert!(!S_FALSE.swap(true, SeqCst));
    assert!(S_TRUE.swap(false, SeqCst));
    assert!(S_INT.fetch_add(1, SeqCst) == 0);
    assert!(S_UINT.fetch_add(1, SeqCst) == 0);
}

#[test]
fn atomic_access_bool() {
    static mut ATOMIC: AtomicBool = AtomicBool::new(false);

    unsafe {
        assert_eq!(*ATOMIC.get_mut(), false);
        ATOMIC.store(true, SeqCst);
        assert_eq!(*ATOMIC.get_mut(), true);
        ATOMIC.fetch_or(false, SeqCst);
        assert_eq!(*ATOMIC.get_mut(), true);
        ATOMIC.fetch_and(false, SeqCst);
        assert_eq!(*ATOMIC.get_mut(), false);
        ATOMIC.fetch_nand(true, SeqCst);
        assert_eq!(*ATOMIC.get_mut(), true);
        ATOMIC.fetch_xor(true, SeqCst);
        assert_eq!(*ATOMIC.get_mut(), false);
    }
}

#[test]
fn atomic_alignment() {
    use std::mem::{align_of, size_of};

    #[cfg(target_has_atomic = "8")]
    assert_eq!(align_of::<AtomicBool>(), size_of::<AtomicBool>());
    #[cfg(target_has_atomic = "ptr")]
    assert_eq!(align_of::<AtomicPtr<u8>>(), size_of::<AtomicPtr<u8>>());
    #[cfg(target_has_atomic = "8")]
    assert_eq!(align_of::<AtomicU8>(), size_of::<AtomicU8>());
    #[cfg(target_has_atomic = "8")]
    assert_eq!(align_of::<AtomicI8>(), size_of::<AtomicI8>());
    #[cfg(target_has_atomic = "16")]
    assert_eq!(align_of::<AtomicU16>(), size_of::<AtomicU16>());
    #[cfg(target_has_atomic = "16")]
    assert_eq!(align_of::<AtomicI16>(), size_of::<AtomicI16>());
    #[cfg(target_has_atomic = "32")]
    assert_eq!(align_of::<AtomicU32>(), size_of::<AtomicU32>());
    #[cfg(target_has_atomic = "32")]
    assert_eq!(align_of::<AtomicI32>(), size_of::<AtomicI32>());
    #[cfg(target_has_atomic = "64")]
    assert_eq!(align_of::<AtomicU64>(), size_of::<AtomicU64>());
    #[cfg(target_has_atomic = "64")]
    assert_eq!(align_of::<AtomicI64>(), size_of::<AtomicI64>());
    #[cfg(target_has_atomic = "128")]
    assert_eq!(align_of::<AtomicU128>(), size_of::<AtomicU128>());
    #[cfg(target_has_atomic = "128")]
    assert_eq!(align_of::<AtomicI128>(), size_of::<AtomicI128>());
    #[cfg(target_has_atomic = "ptr")]
    assert_eq!(align_of::<AtomicUsize>(), size_of::<AtomicUsize>());
    #[cfg(target_has_atomic = "ptr")]
    assert_eq!(align_of::<AtomicIsize>(), size_of::<AtomicIsize>());
}

#[test]
fn atomic_compare_exchange() {
    use Ordering::*;

    static ATOMIC: AtomicIsize = AtomicIsize::new(0);

    ATOMIC.compare_exchange(0, 1, Relaxed, Relaxed).ok();
    ATOMIC.compare_exchange(0, 1, Acquire, Relaxed).ok();
    ATOMIC.compare_exchange(0, 1, Release, Relaxed).ok();
    ATOMIC.compare_exchange(0, 1, AcqRel, Relaxed).ok();
    ATOMIC.compare_exchange(0, 1, SeqCst, Relaxed).ok();
    ATOMIC.compare_exchange(0, 1, Acquire, Acquire).ok();
    ATOMIC.compare_exchange(0, 1, AcqRel, Acquire).ok();
    ATOMIC.compare_exchange(0, 1, SeqCst, Acquire).ok();
    ATOMIC.compare_exchange(0, 1, SeqCst, SeqCst).ok();
    ATOMIC.compare_exchange_weak(0, 1, Relaxed, Relaxed).ok();
    ATOMIC.compare_exchange_weak(0, 1, Acquire, Relaxed).ok();
    ATOMIC.compare_exchange_weak(0, 1, Release, Relaxed).ok();
    ATOMIC.compare_exchange_weak(0, 1, AcqRel, Relaxed).ok();
    ATOMIC.compare_exchange_weak(0, 1, SeqCst, Relaxed).ok();
    ATOMIC.compare_exchange_weak(0, 1, Acquire, Acquire).ok();
    ATOMIC.compare_exchange_weak(0, 1, AcqRel, Acquire).ok();
    ATOMIC.compare_exchange_weak(0, 1, SeqCst, Acquire).ok();
    ATOMIC.compare_exchange_weak(0, 1, SeqCst, SeqCst).ok();
}
