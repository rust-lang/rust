// error-pattern unable to turn pointer into raw bytes
#![feature(const_ptr_read)]
#![feature(const_ptr_offset)]

const C: () = unsafe {
    let foo = Some(&42 as *const i32);
    let one_and_a_half_pointers = std::mem::size_of::<*const i32>()/2*3;
    (&foo as *const _ as *const u8).add(one_and_a_half_pointers).read();
};

fn main() {
}
