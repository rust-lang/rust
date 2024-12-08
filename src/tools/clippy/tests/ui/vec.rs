#![warn(clippy::useless_vec)]
#![allow(clippy::nonstandard_macro_braces, clippy::uninlined_format_args, unused)]

use std::rc::Rc;

struct StructWithVec {
    _x: Vec<i32>,
}

fn on_slice(_: &[u8]) {}

fn on_mut_slice(_: &mut [u8]) {}

#[allow(clippy::ptr_arg)]
fn on_vec(_: &Vec<u8>) {}

fn on_mut_vec(_: &mut Vec<u8>) {}

struct Line {
    length: usize,
}

impl Line {
    fn length(&self) -> usize {
        self.length
    }
}

fn main() {
    on_slice(&vec![]);
    on_slice(&[]);
    on_mut_slice(&mut vec![]);

    on_slice(&vec![1, 2]);
    on_slice(&[1, 2]);
    on_mut_slice(&mut vec![1, 2]);

    on_slice(&vec![1, 2]);
    on_slice(&[1, 2]);
    on_mut_slice(&mut vec![1, 2]);
    #[rustfmt::skip]
    on_slice(&vec!(1, 2));
    on_slice(&[1, 2]);
    on_mut_slice(&mut vec![1, 2]);

    on_slice(&vec![1; 2]);
    on_slice(&[1; 2]);
    on_mut_slice(&mut vec![1; 2]);

    on_vec(&vec![]);
    on_vec(&vec![1, 2]);
    on_vec(&vec![1; 2]);
    on_mut_vec(&mut vec![]);
    on_mut_vec(&mut vec![1, 2]);
    on_mut_vec(&mut vec![1; 2]);

    // Now with non-constant expressions
    let line = Line { length: 2 };

    on_slice(&vec![2; line.length]);
    on_slice(&vec![2; line.length()]);
    on_mut_slice(&mut vec![2; line.length]);
    on_mut_slice(&mut vec![2; line.length()]);

    on_vec(&vec![1; 201]); // Ok, size of `vec` higher than `too_large_for_stack`
    on_mut_vec(&mut vec![1; 201]); // Ok, size of `vec` higher than `too_large_for_stack`

    // Ok
    for a in vec![1; 201] {
        println!("{:?}", a);
    }

    // https://github.com/rust-lang/rust-clippy/issues/2262#issuecomment-783979246
    let _x: i32 = vec![1, 2, 3].iter().sum();

    // Do lint
    let mut x = vec![1, 2, 3];
    x.fill(123);
    dbg!(x[0]);
    dbg!(x.len());
    dbg!(x.iter().sum::<i32>());

    let _x: &[i32] = &vec![1, 2, 3];

    for _ in vec![1, 2, 3] {}

    // Don't lint
    let x = vec![1, 2, 3];
    let _v: Vec<i32> = x;

    let x = vec![1, 2, 3];
    let _s = StructWithVec { _x: x };

    // Explicit type annotation would make the change to [1, 2, 3]
    // a compile error.
    let _x: Vec<i32> = vec![1, 2, 3];

    // Calling a Vec method through a mutable reference
    let mut x = vec![1, 2, 3];
    let re = &mut x;
    re.push(4);

    // Comparing arrays whose length is not equal is a compile error
    let x = vec![1, 2, 3];
    let y = vec![1, 2, 3, 4];
    dbg!(x == y);

    // Non-copy types
    let _x = vec![String::new(); 10];
    #[allow(clippy::rc_clone_in_vec_init)]
    let _x = vec![Rc::new(1); 10];

    // Too large
    let _x = vec![1; 201];
}

fn issue11075() {
    macro_rules! repro {
        ($e:expr) => {
            stringify!($e)
        };
    }
    #[allow(clippy::never_loop)]
    for _string in vec![repro!(true), repro!(null)] {
        unimplemented!();
    }

    macro_rules! in_macro {
        ($e:expr, $vec:expr, $vec2:expr) => {{
            vec![1; 2].fill(3);
            vec![1, 2].fill(3);
            for _ in vec![1, 2] {}
            for _ in vec![1; 2] {}
            for _ in vec![$e, $e] {}
            for _ in vec![$e; 2] {}
            for _ in $vec {}
            for _ in $vec2 {}
        }};
    }

    in_macro!(1, vec![1, 2], vec![1; 2]);

    macro_rules! from_macro {
        () => {
            vec![1, 2, 3]
        };
    }
    macro_rules! from_macro_repeat {
        () => {
            vec![1; 3]
        };
    }

    for _ in from_macro!() {}
    for _ in from_macro_repeat!() {}
}

#[clippy::msrv = "1.53"]
fn above() {
    for a in vec![1, 2, 3] {
        let _: usize = a;
    }

    for a in vec![String::new(), String::new()] {
        let _: String = a;
    }
}

#[clippy::msrv = "1.52"]
fn below() {
    for a in vec![1, 2, 3] {
        let _: usize = a;
    }

    for a in vec![String::new(), String::new()] {
        let _: String = a;
    }
}

fn func_needing_vec(_bar: usize, _baz: Vec<usize>) {}
fn func_not_needing_vec(_bar: usize, _baz: usize) {}

fn issue11861() {
    macro_rules! this_macro_needs_vec {
        ($x:expr) => {{
            func_needing_vec($x.iter().sum(), $x);
            for _ in $x {}
        }};
    }
    macro_rules! this_macro_doesnt_need_vec {
        ($x:expr) => {{ func_not_needing_vec($x.iter().sum(), $x.iter().sum()) }};
    }

    // Do not lint the next line
    this_macro_needs_vec!(vec![1]);
    this_macro_doesnt_need_vec!(vec![1]); //~ ERROR: useless use of `vec!`

    macro_rules! m {
        ($x:expr) => {
            fn f2() {
                let _x: Vec<i32> = $x;
            }
            fn f() {
                let _x = $x;
                $x.starts_with(&[]);
            }
        };
    }

    // should not lint
    m!(vec![1]);
}

fn issue_11958() {
    fn f(_s: &[String]) {}

    // should not lint, `String` is not `Copy`
    f(&vec!["test".to_owned(); 2]);
}

fn issue_12101() {
    for a in &(vec![1, 2]) {}
}
