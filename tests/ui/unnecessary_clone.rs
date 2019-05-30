#![warn(clippy::clone_on_ref_ptr)]
#![allow(unused)]

use std::cell::RefCell;
use std::collections::HashSet;
use std::collections::VecDeque;
use std::rc::{self, Rc};
use std::sync::{self, Arc};

trait SomeTrait {}
struct SomeImpl;
impl SomeTrait for SomeImpl {}

fn main() {}

fn clone_on_copy() {
    42.clone();

    vec![1].clone(); // ok, not a Copy type
    Some(vec![1]).clone(); // ok, not a Copy type
    (&42).clone();

    let rc = RefCell::new(0);
    rc.borrow().clone();
}

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

fn clone_on_double_ref() {
    let x = vec![1];
    let y = &&x;
    let z: &Vec<_> = y.clone();

    println!("{:p} {:p}", *y, z);
}

fn iter_clone_collect() {
    let v = [1, 2, 3, 4, 5];
    let v2: Vec<isize> = v.iter().cloned().collect();
    let v3: HashSet<isize> = v.iter().cloned().collect();
    let v4: VecDeque<isize> = v.iter().cloned().collect();

    // Handle macro expansion in suggestion
    let _: Vec<isize> = vec![1, 2, 3].iter().cloned().collect();

    // Issue #3704
    unsafe {
        let _: Vec<u8> = std::ffi::CStr::from_ptr(std::ptr::null())
            .to_bytes()
            .iter()
            .cloned()
            .collect();
    }
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
