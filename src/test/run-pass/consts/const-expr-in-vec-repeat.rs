// run-pass
// Check that constant expressions can be used in vec repeat syntax.

// pretty-expanded FIXME #23616

pub fn main() {

    const FOO: usize = 2;
    let _v = [0; FOO*3*2/2];

}
