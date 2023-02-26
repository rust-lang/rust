#![allow(unused_labels)]

// FIXME(#108019): outdated Unicode table
// fn foo() {
//     '🥺 loop {
//         break
//     }
// }

fn bar() {
    '🐱 loop {
    //~^ ERROR labeled expression must be followed by `:`
    //~| ERROR lifetimes or labels cannot contain emojis
        break
    }
}

fn qux() {
    'a🐱 loop {
    //~^ ERROR labeled expression must be followed by `:`
    //~| ERROR lifetimes or labels cannot contain emojis
        break
    }
}

fn quux() {
    '1🐱 loop {
    //~^ ERROR labeled expression must be followed by `:`
    //~| ERROR lifetimes or labels cannot start with a number
        break
    }
}

fn x<'🐱>() -> &'🐱 () {
    //~^ ERROR lifetimes or labels cannot contain emojis
    //~| ERROR lifetimes or labels cannot contain emojis
    &()
}

fn y() {
    'a🐱: loop {}
    //~^ ERROR lifetimes or labels cannot contain emojis
}

fn main() {}
