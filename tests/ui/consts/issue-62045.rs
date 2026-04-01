//@ check-pass

fn main() {
    assert_eq!(&mut [0; 1][..], &mut []);
}
