use std::mem;

fn main() {
    let mut arr = [1, 2, 3];

    let i = 0usize;
    let j = 1usize;

    mem::swap(
        &mut arr[i],
        &mut arr[j], //~ ERROR cannot borrow `arr[_]` as mutable more than once at a time
    );
}
