// Issue #7988
// Transmuting non-immediate type to immediate type

// pretty-expanded FIXME #23616

pub fn main() {
    unsafe {
        ::std::mem::transmute::<[isize; 1],isize>([1])
    };
}
