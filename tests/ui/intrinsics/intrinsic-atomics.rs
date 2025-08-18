//@ run-pass
#![feature(core_intrinsics)]
use std::intrinsics::{self as rusti, AtomicOrdering::*};

pub fn main() {
    unsafe {
        let mut x: Box<_> = Box::new(1);

        assert_eq!(rusti::atomic_load::<_, { SeqCst }>(&*x), 1);
        *x = 5;
        assert_eq!(rusti::atomic_load::<_, { Acquire }>(&*x), 5);

        rusti::atomic_store::<_, { SeqCst }>(&mut *x, 3);
        assert_eq!(*x, 3);
        rusti::atomic_store::<_, { Release }>(&mut *x, 1);
        assert_eq!(*x, 1);

        assert_eq!(rusti::atomic_cxchg::<_, { SeqCst }, { SeqCst }>(&mut *x, 1, 2), (1, true));
        assert_eq!(*x, 2);

        assert_eq!(rusti::atomic_cxchg::<_, { Acquire }, { Acquire }>(&mut *x, 1, 3), (2, false));
        assert_eq!(*x, 2);

        assert_eq!(rusti::atomic_cxchg::<_, { Release }, { Relaxed }>(&mut *x, 2, 1), (2, true));
        assert_eq!(*x, 1);

        assert_eq!(rusti::atomic_xchg::<_, { SeqCst }>(&mut *x, 0), 1);
        assert_eq!(*x, 0);

        assert_eq!(rusti::atomic_xchg::<_, { Acquire }>(&mut *x, 1), 0);
        assert_eq!(*x, 1);

        assert_eq!(rusti::atomic_xchg::<_, { Release }>(&mut *x, 0), 1);
        assert_eq!(*x, 0);

        assert_eq!(rusti::atomic_xadd::<_, _, { SeqCst }>(&mut *x, 1), 0);
        assert_eq!(rusti::atomic_xadd::<_, _, { Acquire }>(&mut *x, 1), 1);
        assert_eq!(rusti::atomic_xadd::<_, _, { Release }>(&mut *x, 1), 2);
        assert_eq!(*x, 3);

        assert_eq!(rusti::atomic_xsub::<_, _, { SeqCst }>(&mut *x, 1), 3);
        assert_eq!(rusti::atomic_xsub::<_, _, { Acquire }>(&mut *x, 1), 2);
        assert_eq!(rusti::atomic_xsub::<_, _, { Release }>(&mut *x, 1), 1);
        assert_eq!(*x, 0);

        loop {
            let res = rusti::atomic_cxchgweak::<_, { SeqCst }, { SeqCst }>(&mut *x, 0, 1);
            assert_eq!(res.0, 0);
            if res.1 {
                break;
            }
        }
        assert_eq!(*x, 1);

        loop {
            let res = rusti::atomic_cxchgweak::<_, { Acquire }, { Acquire }>(&mut *x, 1, 2);
            assert_eq!(res.0, 1);
            if res.1 {
                break;
            }
        }
        assert_eq!(*x, 2);

        loop {
            let res = rusti::atomic_cxchgweak::<_, { Release }, { Relaxed }>(&mut *x, 2, 3);
            assert_eq!(res.0, 2);
            if res.1 {
                break;
            }
        }
        assert_eq!(*x, 3);
    }
}
