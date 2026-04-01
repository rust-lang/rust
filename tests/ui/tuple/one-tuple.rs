//@ run-pass
// Why one-tuples? Because macros.


pub fn main() {
    match ('c',) {
        (x,) => {
            assert_eq!(x, 'c');
        }
    }
    // test the 1-tuple type too
    let x: (char,) = ('d',);
    let (y,) = x;
    assert_eq!(y, 'd');
}
