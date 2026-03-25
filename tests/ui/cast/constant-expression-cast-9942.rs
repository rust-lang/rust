// https://github.com/rust-lang/rust/issues/9942
//@ build-pass

pub fn main() {
    const S: usize = 23 as usize; [0; S]; ()
}
