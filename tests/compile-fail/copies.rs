#![feature(plugin)]
#![plugin(clippy)]

#![allow(dead_code, no_effect)]
#![allow(let_and_return)]
#![allow(needless_return)]
#![allow(unused_variables)]

fn foo() -> bool { unimplemented!() }

#[deny(if_same_then_else)]
fn if_same_then_else() -> &'static str {
    if true {
        foo();
    }
    else { //~ERROR this if has identical blocks
        foo();
    }

    if true {
        foo();
        foo();
    }
    else {
        foo();
    }

    let _ = if true {
        foo();
        42
    }
    else { //~ERROR this if has identical blocks
        foo();
        42
    };

    if true {
        foo();
    }

    let _ = if true {
        42
    }
    else { //~ERROR this if has identical blocks
        42
    };

    if true {
        let bar = if true {
            42
        }
        else {
            43
        };

        while foo() { break; }
        bar + 1;
    }
    else { //~ERROR this if has identical blocks
        let bar = if true {
            42
        }
        else {
            43
        };

        while foo() { break; }
        bar + 1;
    }

    if true {
        match 42 {
            42 => (),
            a if a > 0 => (),
            10...15 => (),
            _ => (),
        }
    }
    else if false {
        foo();
    }
    else if foo() { //~ERROR this if has identical blocks
        match 42 {
            42 => (),
            a if a > 0 => (),
            10...15 => (),
            _ => (),
        }
    }

    if true {
        if let Some(a) = Some(42) {}
    }
    else { //~ERROR this if has identical blocks
        if let Some(a) = Some(42) {}
    }

    if true {
        if let Some(a) = Some(42) {}
    }
    else {
        if let Some(a) = Some(43) {}
    }

    if true {
        let foo = "";
        return &foo[0..];
    }
    else if false {
        let foo = "bar";
        return &foo[0..];
    }
    else { //~ERROR this if has identical blocks
        let foo = "";
        return &foo[0..];
    }
}

#[deny(ifs_same_cond)]
#[allow(if_same_then_else)] // all empty blocks
fn ifs_same_cond() {
    let a = 0;
    let b = false;

    if b {
    }
    else if b { //~ERROR this if has the same condition as a previous if
    }

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
