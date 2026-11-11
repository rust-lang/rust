//@ run-pass
// Check that constant expressions can be used for declaring the
// type of a fixed length vector.


pub fn main() {

    const FOO: usize = 2;
    let _v: [isize; FOO*3];

}
