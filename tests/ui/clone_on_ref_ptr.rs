#![warn(clippy::clone_on_ref_ptr)]

use std::rc::{Rc, Weak as RcWeak};
use std::sync::{Arc, Weak as ArcWeak};

fn main() {}

fn clone_on_ref_ptr(rc: Rc<str>, rc_weak: RcWeak<str>, arc: Arc<str>, arc_weak: ArcWeak<str>) {
    rc.clone();
    //~^ clone_on_ref_ptr
    rc_weak.clone();
    //~^ clone_on_ref_ptr
    arc.clone();
    //~^ clone_on_ref_ptr
    arc_weak.clone();
    //~^ clone_on_ref_ptr

    Rc::clone(&rc);
    Arc::clone(&arc);
    RcWeak::clone(&rc_weak);
    ArcWeak::clone(&arc_weak);
}

trait SomeTrait {}
struct SomeImpl;
impl SomeTrait for SomeImpl {}

fn trait_object() {
    let x = Arc::new(SomeImpl);
    let _: Arc<dyn SomeTrait> = x.clone();
    //~^ clone_on_ref_ptr
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
        //~^ clone_on_ref_ptr
    }
}
