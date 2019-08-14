// run-pass
#![feature(box_syntax)]
#![feature(intrinsics)]

mod rusti {
    extern "rust-intrinsic" {
        pub fn atomic_cxchg<T>(dst: *mut T, old: T, src: T) -> (T, bool);
        pub fn atomic_cxchg_acq<T>(dst: *mut T, old: T, src: T) -> (T, bool);
        pub fn atomic_cxchg_rel<T>(dst: *mut T, old: T, src: T) -> (T, bool);

        pub fn atomic_cxchgweak<T>(dst: *mut T, old: T, src: T) -> (T, bool);
        pub fn atomic_cxchgweak_acq<T>(dst: *mut T, old: T, src: T) -> (T, bool);
        pub fn atomic_cxchgweak_rel<T>(dst: *mut T, old: T, src: T) -> (T, bool);

        pub fn atomic_load<T>(src: *const T) -> T;
        pub fn atomic_load_acq<T>(src: *const T) -> T;

        pub fn atomic_store<T>(dst: *mut T, val: T);
        pub fn atomic_store_rel<T>(dst: *mut T, val: T);

        pub fn atomic_xchg<T>(dst: *mut T, src: T) -> T;
        pub fn atomic_xchg_acq<T>(dst: *mut T, src: T) -> T;
        pub fn atomic_xchg_rel<T>(dst: *mut T, src: T) -> T;

        pub fn atomic_xadd<T>(dst: *mut T, src: T) -> T;
        pub fn atomic_xadd_acq<T>(dst: *mut T, src: T) -> T;
        pub fn atomic_xadd_rel<T>(dst: *mut T, src: T) -> T;

        pub fn atomic_xsub<T>(dst: *mut T, src: T) -> T;
        pub fn atomic_xsub_acq<T>(dst: *mut T, src: T) -> T;
        pub fn atomic_xsub_rel<T>(dst: *mut T, src: T) -> T;
    }
}

pub fn main() {
    unsafe {
        let mut x: Box<_> = box 1;

        assert_eq!(rusti::atomic_load(&*x), 1);
        *x = 5;
        assert_eq!(rusti::atomic_load_acq(&*x), 5);

        rusti::atomic_store(&mut *x,3);
        assert_eq!(*x, 3);
        rusti::atomic_store_rel(&mut *x,1);
        assert_eq!(*x, 1);

        assert_eq!(rusti::atomic_cxchg(&mut *x, 1, 2), (1, true));
        assert_eq!(*x, 2);

        assert_eq!(rusti::atomic_cxchg_acq(&mut *x, 1, 3), (2, false));
        assert_eq!(*x, 2);

        assert_eq!(rusti::atomic_cxchg_rel(&mut *x, 2, 1), (2, true));
        assert_eq!(*x, 1);

        assert_eq!(rusti::atomic_xchg(&mut *x, 0), 1);
        assert_eq!(*x, 0);

        assert_eq!(rusti::atomic_xchg_acq(&mut *x, 1), 0);
        assert_eq!(*x, 1);

        assert_eq!(rusti::atomic_xchg_rel(&mut *x, 0), 1);
        assert_eq!(*x, 0);

        assert_eq!(rusti::atomic_xadd(&mut *x, 1), 0);
        assert_eq!(rusti::atomic_xadd_acq(&mut *x, 1), 1);
        assert_eq!(rusti::atomic_xadd_rel(&mut *x, 1), 2);
        assert_eq!(*x, 3);

        assert_eq!(rusti::atomic_xsub(&mut *x, 1), 3);
        assert_eq!(rusti::atomic_xsub_acq(&mut *x, 1), 2);
        assert_eq!(rusti::atomic_xsub_rel(&mut *x, 1), 1);
        assert_eq!(*x, 0);

        loop {
            let res = rusti::atomic_cxchgweak(&mut *x, 0, 1);
            assert_eq!(res.0, 0);
            if res.1 {
                break;
            }
        }
        assert_eq!(*x, 1);

        loop {
            let res = rusti::atomic_cxchgweak_acq(&mut *x, 1, 2);
            assert_eq!(res.0, 1);
            if res.1 {
                break;
            }
        }
        assert_eq!(*x, 2);

        loop {
            let res = rusti::atomic_cxchgweak_rel(&mut *x, 2, 3);
            assert_eq!(res.0, 2);
            if res.1 {
                break;
            }
        }
        assert_eq!(*x, 3);
    }
}
