//@ run-pass
// Issue #7988
// Transmuting non-immediate type to immediate type


pub fn main() {
    unsafe {
        ::std::mem::transmute::<[isize; 1],isize>([1])
    };
}
