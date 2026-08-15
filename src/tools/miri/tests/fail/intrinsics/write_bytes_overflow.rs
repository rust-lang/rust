use std::mem;

fn main() {
    let mut y = 0;
    unsafe {
        (&mut y as *mut i32).write_bytes(0u8, 1usize << (mem::size_of::<usize>() * 8 - 1));
        //~^ ERROR: overflow computing total size of `write_bytes`
    }
}
