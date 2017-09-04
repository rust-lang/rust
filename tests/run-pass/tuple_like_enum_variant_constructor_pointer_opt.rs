// FIXME: investigate again once #296 is fixed
// compile-flags: -Zmir-emit-validate=0

fn main() {
    let x = 5;
    assert_eq!(Some(&x).map(Some), Some(Some(&x)));
}
