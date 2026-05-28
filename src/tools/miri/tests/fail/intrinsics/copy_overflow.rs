use std::mem;

fn main() {
    let x = 0;
    let mut y = 0;
    unsafe {
        (&mut y as *mut i32).copy_from(&x, 1usize << (mem::size_of::<usize>() * 8 - 1));
        //~^ERROR: overflow computing total size
    }
}
