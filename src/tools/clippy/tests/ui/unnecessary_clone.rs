// does not test any rustfixable lints
#![warn(clippy::clone_on_ref_ptr)]
#![allow(unused)]
#![allow(clippy::redundant_clone, clippy::uninlined_format_args, clippy::unnecessary_wraps)]

use std::cell::RefCell;
use std::rc::{self, Rc};
use std::sync::{self, Arc};

trait SomeTrait {}
struct SomeImpl;
impl SomeTrait for SomeImpl {}

fn main() {}

fn clone_on_ref_ptr() {
    let rc = Rc::new(true);
    let arc = Arc::new(true);

    let rcweak = Rc::downgrade(&rc);
    let arc_weak = Arc::downgrade(&arc);

    rc.clone();
    Rc::clone(&rc);

    arc.clone();
    Arc::clone(&arc);

    rcweak.clone();
    rc::Weak::clone(&rcweak);

    arc_weak.clone();
    sync::Weak::clone(&arc_weak);

    let x = Arc::new(SomeImpl);
    let _: Arc<dyn SomeTrait> = x.clone();
}

fn clone_on_copy_generic<T: Copy>(t: T) {
    t.clone();

    Some(t).clone();
}

mod many_derefs {
    struct A;
    struct B;
    struct C;
    struct D;
    #[derive(Copy, Clone)]
    struct E;

    macro_rules! impl_deref {
        ($src:ident, $dst:ident) => {
            impl std::ops::Deref for $src {
                type Target = $dst;
                fn deref(&self) -> &Self::Target {
                    &$dst
                }
            }
        };
    }

    impl_deref!(A, B);
    impl_deref!(B, C);
    impl_deref!(C, D);
    impl std::ops::Deref for D {
        type Target = &'static E;
        fn deref(&self) -> &Self::Target {
            &&E
        }
    }

    fn go1() {
        let a = A;
        let _: E = a.clone();
        let _: E = *****a;
    }
}

mod issue2076 {
    use std::rc::Rc;

    macro_rules! try_opt {
        ($expr: expr) => {
            match $expr {
                Some(value) => value,
                None => return None,
            }
        };
    }

    fn func() -> Option<Rc<u8>> {
        let rc = Rc::new(42);
        Some(try_opt!(Some(rc)).clone())
    }
}
