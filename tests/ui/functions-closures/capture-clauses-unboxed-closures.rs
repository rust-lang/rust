//@ run-pass
fn each<'a,T,F:FnMut(&'a T)>(x: &'a [T], mut f: F) {
    for val in x {
        f(val)
    }
}

fn main() {
    let mut sum = 0;
    let elems = [ 1, 2, 3, 4, 5 ];
    each(&elems, |val: &usize| sum += *val);
    assert_eq!(sum, 15);
}
