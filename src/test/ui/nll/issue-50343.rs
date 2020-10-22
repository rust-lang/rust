// run-pass

#![deny(unused_mut)]

fn main() {
    vec![42].iter().map(drop).for_each(drop);
    vec![(42, 22)].iter().map(|(_x, _y)| ()).for_each(drop);
}
