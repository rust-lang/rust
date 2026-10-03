//@ run-rustfix
#[allow(unused_mut)]
fn test() {
    let w: &mut [isize];
    w[5] = 0; //~ ERROR [E0381]

    let mut w: &mut [isize];
    w[5] = 0; //~ ERROR [E0381]
}

fn main() { test(); }
