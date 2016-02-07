#![feature(plugin)]
#![plugin(clippy)]

#![allow(dead_code)]
#![allow(let_and_return)]
#![allow(needless_return)]
#![allow(unused_variables)]
#![deny(if_same_then_else)]
#![deny(ifs_same_cond)]

fn foo() -> bool { unimplemented!() }

fn if_same_then_else() -> &'static str {
    if true { //~ERROR this if has the same then and else blocks
        foo();
    }
    else {
        foo();
    }

    if true {
        foo();
        foo();
    }
    else {
        foo();
    }

    let _ = if true { //~ERROR this if has the same then and else blocks
        foo();
        42
    }
    else {
        foo();
        42
    };

    if true {
        foo();
    }

    let _ = if true { //~ERROR this if has the same then and else blocks
        42
    }
    else {
        42
    };

    if true { //~ERROR this if has the same then and else blocks
        let bar = if true {
            42
        }
        else {
            43
        };

        while foo() { break; }
        bar + 1;
    }
    else {
        let bar = if true {
            42
        }
        else {
            43
        };

        while foo() { break; }
        bar + 1;
    }

    if true { //~ERROR this if has the same then and else blocks
        match 42 {
            42 => (),
            a if a > 0 => (),
            10...15 => (),
            _ => (),
        }
    }
    else {
        match 42 {
            42 => (),
            a if a > 0 => (),
            10...15 => (),
            _ => (),
        }
    }

    if true { //~ERROR this if has the same then and else blocks
        if let Some(a) = Some(42) {}
    }
    else {
        if let Some(a) = Some(42) {}
    }

    if true { //~ERROR this if has the same then and else blocks
        let foo = "";
        return &foo[0..];
    }
    else {
        let foo = "";
        return &foo[0..];
    }
}

fn ifs_same_cond() {
    let a = 0;

    if a == 1 {
    }
    else if a == 1 { //~ERROR this if has the same condition as a previous if
    }

    if 2*a == 1 {
    }
    else if 2*a == 2 {
    }
    else if 2*a == 1 { //~ERROR this if has the same condition as a previous if
    }
    else if a == 1 {
    }

    let mut v = vec![1];
    if v.pop() == None { // ok, functions
    }
    else if v.pop() == None {
    }

    if v.len() == 42 { // ok, functions
    }
    else if v.len() == 42 {
    }
}

fn main() {}
