//@run-rustfix
#![warn(clippy::needless_for_each)]
#![allow(unused)]
#![allow(
    clippy::let_unit_value,
    clippy::match_single_binding,
    clippy::needless_return,
    clippy::uninlined_format_args
)]

use std::collections::HashMap;

fn should_lint() {
    let v: Vec<i32> = Vec::new();
    let mut acc = 0;
    v.iter().for_each(|elem| {
        acc += elem;
    });
    v.into_iter().for_each(|elem| {
        acc += elem;
    });

    [1, 2, 3].iter().for_each(|elem| {
        acc += elem;
    });

    let mut hash_map: HashMap<i32, i32> = HashMap::new();
    hash_map.iter().for_each(|(k, v)| {
        acc += k + v;
    });
    hash_map.iter_mut().for_each(|(k, v)| {
        acc += *k + *v;
    });
    hash_map.keys().for_each(|k| {
        acc += k;
    });
    hash_map.values().for_each(|v| {
        acc += v;
    });

    fn my_vec() -> Vec<i32> {
        Vec::new()
    }
    my_vec().iter().for_each(|elem| {
        acc += elem;
    });
}

fn should_not_lint() {
    let v: Vec<i32> = Vec::new();
    let mut acc = 0;

    // `for_each` argument is not closure.
    fn print(x: &i32) {
        println!("{}", x);
    }
    v.iter().for_each(print);

    // User defined type.
    struct MyStruct {
        v: Vec<i32>,
    }
    impl MyStruct {
        fn iter(&self) -> impl Iterator<Item = &i32> {
            self.v.iter()
        }
    }
    let s = MyStruct { v: Vec::new() };
    s.iter().for_each(|elem| {
        acc += elem;
    });

    // `for_each` follows long iterator chain.
    v.iter().chain(v.iter()).for_each(|v| {
        acc += v;
    });
    v.as_slice().iter().for_each(|v| {
        acc += v;
    });
    s.v.iter().for_each(|v| {
        acc += v;
    });

    // `return` is used in `Loop` of the closure.
    v.iter().for_each(|v| {
        for i in 0..*v {
            if i == 10 {
                return;
            } else {
                println!("{}", v);
            }
        }
        if *v == 20 {
            return;
        } else {
            println!("{}", v);
        }
    });

    // Previously transformed iterator variable.
    let it = v.iter();
    it.chain(v.iter()).for_each(|elem| {
        acc += elem;
    });

    // `for_each` is not directly in a statement.
    match 1 {
        _ => v.iter().for_each(|elem| {
            acc += elem;
        }),
    }

    // `for_each` is in a let bingind.
    let _ = v.iter().for_each(|elem| {
        acc += elem;
    });
}

fn main() {}
