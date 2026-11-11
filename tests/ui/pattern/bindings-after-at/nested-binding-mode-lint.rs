//@ check-pass

#![deny(unused_mut)]

fn main() {
    let mut is_mut @ not_mut = 42;
    &mut is_mut;
    &not_mut;
    let not_mut @ mut is_mut = 42;
    &mut is_mut;
    &not_mut;
}
