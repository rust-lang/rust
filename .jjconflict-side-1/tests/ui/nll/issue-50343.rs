//@ run-pass

#![deny(unused_mut)]

fn main() {
    vec![42].iter().map(drop).count();
    vec![(42, 22)].iter().map(|(_x, _y)| ()).count();
}
