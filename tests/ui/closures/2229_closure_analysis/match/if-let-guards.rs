// Check the if let guards don't force capture by value
//@ revisions: e2018 e2021
//@ check-pass
//@[e2018] edition:2018
//@[e2021] edition:2021

#![feature(if_let_guard)]
#![allow(irrefutable_let_patterns)]

fn if_let_underscore(value: Box<E>) {
    |x: &E| {
        match &x {
            E::Number(_) if let _ = *value => { }
            _ => {}
        }
    };
    let x = value;
}

fn if_let_copy(value: Box<E>) {
    |x: &E| {
        match &x {
            E::Number(_) if let E::Number(n) = *value => { }
            _ => {}
        }
    };
    let x = value;
}

fn if_let_ref(value: Box<E>) {
    |x: &E| {
        match &x {
            E::Number(_) if let E::Number(ref n) = *value => { }
            _ => {}
        }
    };
    let x = value;
}

fn if_let_ref_mut(mut value: Box<E>) {
    |x: &E| {
        match &x {
            E::Number(_) if let E::Number(ref mut n) = *value => { }
            _ => {}
        }
    };
    let x = value;
}

enum E {
    String(String),
    Number(i32),
}

fn main() {}
