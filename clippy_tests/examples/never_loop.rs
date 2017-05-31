#![feature(plugin)]
#![plugin(clippy)]

#![warn(never_loop)]
#![allow(single_match, while_true)]

fn break_stmt() {
    loop {
        break;
    }
}

fn conditional_break() {
    let mut x = 5;
    loop {
        x -= 1;
        if x == 1 {
            break
        }
    }
}

fn nested_loop() {
    loop {
        while true {
            break
        }
        break
    }
}

fn if_false() {
    let x = 1;
    loop {
        if x == 1 {
            return
        }
    }
}

fn match_false() {
    let x = 1;
    loop {
        match x {
            1 => return,
            _ => (),
        }
    }
}

fn main() {
    break_stmt();
    conditional_break();
    nested_loop();
    if_false();
    match_false();
}
