//@ run-pass
#![feature(intrinsics)]

mod rusti {

    #[rustc_intrinsic]
    pub unsafe fn atomic_cxchg_seqcst_seqcst<T>(dst: *mut T, old: T, src: T) -> (T, bool);
    #[rustc_intrinsic]
    pub unsafe fn atomic_cxchg_acquire_acquire<T>(dst: *mut T, old: T, src: T) -> (T, bool);
    #[rustc_intrinsic]
    pub unsafe fn atomic_cxchg_release_relaxed<T>(dst: *mut T, old: T, src: T) -> (T, bool);

    #[rustc_intrinsic]
    pub unsafe fn atomic_cxchgweak_seqcst_seqcst<T>(dst: *mut T, old: T, src: T) -> (T, bool);
    #[rustc_intrinsic]
    pub unsafe fn atomic_cxchgweak_acquire_acquire<T>(dst: *mut T, old: T, src: T) -> (T, bool);
    #[rustc_intrinsic]
    pub unsafe fn atomic_cxchgweak_release_relaxed<T>(dst: *mut T, old: T, src: T) -> (T, bool);

    #[rustc_intrinsic]
    pub unsafe fn atomic_load_seqcst<T>(src: *const T) -> T;
    #[rustc_intrinsic]
    pub unsafe fn atomic_load_acquire<T>(src: *const T) -> T;

    #[rustc_intrinsic]
    pub unsafe fn atomic_store_seqcst<T>(dst: *mut T, val: T);
    #[rustc_intrinsic]
    pub unsafe fn atomic_store_release<T>(dst: *mut T, val: T);

    #[rustc_intrinsic]
    pub unsafe fn atomic_xchg_seqcst<T>(dst: *mut T, src: T) -> T;
    #[rustc_intrinsic]
    pub unsafe fn atomic_xchg_acquire<T>(dst: *mut T, src: T) -> T;
    #[rustc_intrinsic]
    pub unsafe fn atomic_xchg_release<T>(dst: *mut T, src: T) -> T;

    #[rustc_intrinsic]
    pub unsafe fn atomic_xadd_seqcst<T>(dst: *mut T, src: T) -> T;
    #[rustc_intrinsic]
    pub unsafe fn atomic_xadd_acquire<T>(dst: *mut T, src: T) -> T;
    #[rustc_intrinsic]
    pub unsafe fn atomic_xadd_release<T>(dst: *mut T, src: T) -> T;

    #[rustc_intrinsic]
    pub unsafe fn atomic_xsub_seqcst<T>(dst: *mut T, src: T) -> T;
    #[rustc_intrinsic]
    pub unsafe fn atomic_xsub_acquire<T>(dst: *mut T, src: T) -> T;
    #[rustc_intrinsic]
    pub unsafe fn atomic_xsub_release<T>(dst: *mut T, src: T) -> T;
}

pub fn main() {
    unsafe {
        let mut x: Box<_> = Box::new(1);

        assert_eq!(rusti::atomic_load_seqcst(&*x), 1);
        *x = 5;
        assert_eq!(rusti::atomic_load_acquire(&*x), 5);

        rusti::atomic_store_seqcst(&mut *x, 3);
        assert_eq!(*x, 3);
        rusti::atomic_store_release(&mut *x, 1);
        assert_eq!(*x, 1);

        assert_eq!(rusti::atomic_cxchg_seqcst_seqcst(&mut *x, 1, 2), (1, true));
        assert_eq!(*x, 2);

        assert_eq!(rusti::atomic_cxchg_acquire_acquire(&mut *x, 1, 3), (2, false));
        assert_eq!(*x, 2);

        assert_eq!(rusti::atomic_cxchg_release_relaxed(&mut *x, 2, 1), (2, true));
        assert_eq!(*x, 1);

        assert_eq!(rusti::atomic_xchg_seqcst(&mut *x, 0), 1);
        assert_eq!(*x, 0);

        assert_eq!(rusti::atomic_xchg_acquire(&mut *x, 1), 0);
        assert_eq!(*x, 1);

        assert_eq!(rusti::atomic_xchg_release(&mut *x, 0), 1);
        assert_eq!(*x, 0);

        assert_eq!(rusti::atomic_xadd_seqcst(&mut *x, 1), 0);
        assert_eq!(rusti::atomic_xadd_acquire(&mut *x, 1), 1);
        assert_eq!(rusti::atomic_xadd_release(&mut *x, 1), 2);
        assert_eq!(*x, 3);

        assert_eq!(rusti::atomic_xsub_seqcst(&mut *x, 1), 3);
        assert_eq!(rusti::atomic_xsub_acquire(&mut *x, 1), 2);
        assert_eq!(rusti::atomic_xsub_release(&mut *x, 1), 1);
        assert_eq!(*x, 0);

        loop {
            let res = rusti::atomic_cxchgweak_seqcst_seqcst(&mut *x, 0, 1);
            assert_eq!(res.0, 0);
            if res.1 {
                break;
            }
        }
        assert_eq!(*x, 1);

        loop {
            let res = rusti::atomic_cxchgweak_acquire_acquire(&mut *x, 1, 2);
            assert_eq!(res.0, 1);
            if res.1 {
                break;
            }
        }
        assert_eq!(*x, 2);

        loop {
            let res = rusti::atomic_cxchgweak_release_relaxed(&mut *x, 2, 3);
            assert_eq!(res.0, 2);
            if res.1 {
                break;
            }
        }
        assert_eq!(*x, 3);
    }
}
