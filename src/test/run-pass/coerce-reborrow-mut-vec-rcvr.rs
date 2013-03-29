trait Reverser {
    fn reverse(&self);
}

impl<'self> Reverser for &'self mut [uint] {
    fn reverse(&self) {
        vec::reverse(*self);
    }
}

fn bar(v: &mut [uint]) {
    v.reverse();
    v.reverse();
    v.reverse();
}

pub fn main() {
    let mut the_vec = ~[1, 2, 3, 100];
    bar(the_vec);
    assert!(the_vec == ~[100, 3, 2, 1]);
}
