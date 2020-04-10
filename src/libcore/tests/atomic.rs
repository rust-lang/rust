use core::sync::atomic::Ordering::SeqCst;
use core::sync::atomic::*;

#[test]
fn bool_() {
    let a = AtomicBool::new(false);
    assert_eq!(a.compare_and_swap(false, true, SeqCst), false);
    assert_eq!(a.compare_and_swap(false, true, SeqCst), true);

    a.store(false, SeqCst);
    assert_eq!(a.compare_and_swap(false, true, SeqCst), false);
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

#[test]
fn atomic_ptr() {
    // This test assumes a contiguous memory layout for a (tuple) pair of usize
    unsafe {
        let mut mem: (usize, usize) = (1, 2);
        let mut ptr = &mut mem.0 as *mut usize;
        // ptr points to .0
        let atomic = AtomicPtr::new(ptr);
        // atomic points to .0
        assert_eq!(atomic.fetch_add(core::mem::size_of::<usize>(), SeqCst), ptr);
        // atomic points to .1
        ptr = atomic.load(SeqCst);
        // ptr points to .1
        assert_eq!(*ptr, 2);
        atomic.fetch_sub(core::mem::size_of::<usize>(), SeqCst);
        // atomic points to .0
        ptr = atomic.load(SeqCst);
        // ptr points to .0
        assert_eq!(*ptr, 1);

        // now try xor and back
        assert_eq!(atomic.fetch_xor(ptr as usize, SeqCst), ptr);
        // atomic is NULL
        assert_eq!(atomic.fetch_xor(ptr as usize, SeqCst), std::ptr::null_mut());
        // atomic points to .0
        ptr = atomic.load(SeqCst);
        // ptr points to .0
        assert_eq!(*ptr, 1);

        // then and with all 1s
        assert_eq!(atomic.fetch_and(!0, SeqCst), ptr);
        assert_eq!(atomic.load(SeqCst), ptr);

        // then or with all 0s
        assert_eq!(atomic.fetch_or(0, SeqCst), ptr);
        assert_eq!(atomic.load(SeqCst), ptr);

        // then or with all 1s
        assert_eq!(atomic.fetch_or(!0, SeqCst), ptr);
        assert_eq!(atomic.load(SeqCst), !0 as *mut _);

        // then and with all 0s
        assert_eq!(atomic.fetch_and(0, SeqCst), !0 as *mut _);
        assert_eq!(atomic.load(SeqCst), 0 as *mut _);
    }
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
