// Check the if let guards don't force capture by value
//@ revisions: e2018 e2021
//@[e2018] edition:2018
//@[e2021] edition:2021

#![feature(if_let_guard)]
#![allow(irrefutable_let_patterns)]

fn if_let_ref_mut(mut value: Box<E>) {
    let f = |x: &E| {
        match &x {
            E::Number(_) if let E::Number(ref mut n) = *value => { }
            _ => {}
        }
    };
    let x = value;
    //~^ ERROR cannot move out of `value` because it is borrowed
    drop(f);
}

fn if_let_move(value: Box<E>) {
    let f = |x: &E| {
        match &x {
            E::Number(_) if let E::String(s) = *value => { }
            _ => {}
        }
    };
    let x = value;
    //~^ ERROR use of moved value: `value`
}

enum E {
    String(String),
    Number(i32),
}

fn main() {}
