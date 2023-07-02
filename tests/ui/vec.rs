//@run-rustfix
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
