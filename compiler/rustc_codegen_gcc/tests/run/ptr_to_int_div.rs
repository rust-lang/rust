// Compiler:
//
// Run-time:
//   status: 0

use std::hint::black_box;
use std::mem::transmute;

fn main() {
    let pointer = black_box(usize::MAX) as *const ();

    let unsigned = unsafe { transmute::<*const (), usize>(pointer) };
    assert_eq!(unsigned / black_box(2), usize::MAX / 2);
    assert_eq!(unsigned % black_box(2), usize::MAX % 2);

    let signed = unsafe { transmute::<*const (), isize>(pointer) };
    assert_eq!(signed / black_box(2), -1isize / 2);
    assert_eq!(signed % black_box(2), -1isize % 2);
}
