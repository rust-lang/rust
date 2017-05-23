#![feature(plugin)]
#![plugin(clippy)]

#![warn(never_loop)]
#![allow(dead_code, unused)]

fn main() {
    loop {
        println!("This is only ever printed once");
        break;
    }

    let x = 1;
    loop {
        println!("This, too"); // but that's OK
        if x == 1 {
            break;
        }
    }

    loop {
        loop {
            // another one
            break;
        }
        break;
    }

    loop {
        loop {
            if x == 1 { return; }
        }
    }
}
