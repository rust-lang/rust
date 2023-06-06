#![warn(clippy::rc_clone_in_vec_init)]
#![allow(clippy::useless_vec)]
use std::rc::{Rc, Weak as UnSyncWeak};
use std::sync::{Arc, Mutex, Weak as SyncWeak};

fn main() {}

fn should_warn_simple_case() {
    let v = vec![SyncWeak::<u32>::new(); 2];
    let v2 = vec![UnSyncWeak::<u32>::new(); 2];

    let v = vec![Rc::downgrade(&Rc::new("x".to_string())); 2];
    let v = vec![Arc::downgrade(&Arc::new("x".to_string())); 2];
}

fn should_warn_simple_case_with_big_indentation() {
    if true {
        let k = 1;
        dbg!(k);
        if true {
            let v = vec![Arc::downgrade(&Arc::new("x".to_string())); 2];
            let v2 = vec![Rc::downgrade(&Rc::new("x".to_string())); 2];
        }
    }
}

fn should_warn_complex_case() {
    let v = vec![
        Arc::downgrade(&Arc::new(Mutex::new({
            let x = 1;
            dbg!(x);
            x
        })));
        2
    ];

    let v1 = vec![
        Rc::downgrade(&Rc::new(Mutex::new({
            let x = 1;
            dbg!(x);
            x
        })));
        2
    ];
}

fn should_not_warn_custom_weak() {
    #[derive(Clone)]
    struct Weak;

    impl Weak {
        fn new() -> Self {
            Weak
        }
    }

    let v = vec![Weak::new(); 2];
}

fn should_not_warn_vec_from_elem_but_not_weak() {
    let v = vec![String::new(); 2];
    let v1 = vec![1; 2];
    let v2 = vec![
        Box::new(Arc::downgrade(&Arc::new({
            let y = 3;
            dbg!(y);
            y
        })));
        2
    ];
    let v3 = vec![
        Box::new(Rc::downgrade(&Rc::new({
            let y = 3;
            dbg!(y);
            y
        })));
        2
    ];
}

fn should_not_warn_vec_macro_but_not_from_elem() {
    let v = vec![Arc::downgrade(&Arc::new("x".to_string()))];
    let v = vec![Rc::downgrade(&Rc::new("x".to_string()))];
}
