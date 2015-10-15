#![feature(plugin)]
#![feature(convert)]
#![plugin(clippy)]

#![deny(clippy)]

fn test_as_slice() {
    let v = vec![1];
    v.as_slice(); //~ERROR used as_slice() from the 'convert' nightly feature. Use &[..]

    let mut v2 = vec![1];
    v2.as_mut_slice(); //~ERROR used as_mut_slice() from the 'convert' nightly feature. Use &mut [..]
}

fn main() {
    test_as_slice();
}
