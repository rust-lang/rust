#![warn(clippy::rc_clone_in_vec_init)]
#![allow(clippy::useless_vec)]
use std::rc::Rc;
use std::sync::Mutex;

fn main() {}

fn should_warn_simple_case() {
    let v = vec![Rc::new("x".to_string()); 2];
}

fn should_warn_simple_case_with_big_indentation() {
    if true {
        let k = 1;
        dbg!(k);
        if true {
            let v = vec![Rc::new("x".to_string()); 2];
        }
    }
}

fn should_warn_complex_case() {
    let v = vec![
        std::rc::Rc::new(Mutex::new({
            let x = 1;
            dbg!(x);
            x
        }));
        2
    ];

    let v1 = vec![
        Rc::new(Mutex::new({
            let x = 1;
            dbg!(x);
            x
        }));
        2
    ];
}

fn should_not_warn_custom_arc() {
    #[derive(Clone)]
    struct Rc;

    impl Rc {
        fn new() -> Self {
            Rc
        }
    }

    let v = vec![Rc::new(); 2];
}

fn should_not_warn_vec_from_elem_but_not_rc() {
    let v = vec![String::new(); 2];
    let v1 = vec![1; 2];
    let v2 = vec![
        Box::new(std::rc::Rc::new({
            let y = 3;
            dbg!(y);
            y
        }));
        2
    ];
}

fn should_not_warn_vec_macro_but_not_from_elem() {
    let v = vec![Rc::new("x".to_string())];
}
