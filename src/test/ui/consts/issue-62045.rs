// build-pass (FIXME(62277): could be check-pass?)

fn main() {
    assert_eq!(&mut [0; 1][..], &mut []);
}
