// error-pattern unable to turn pointer into raw bytes
// normalize-stderr-test: "alloc[0-9]+\+0x[a-z0-9]+" -> "ALLOC"

const C: () = unsafe {
    let foo = Some(&42 as *const i32);
    let one_and_a_half_pointers = std::mem::size_of::<*const i32>()/2*3;
    (&foo as *const _ as *const u8).add(one_and_a_half_pointers).read();
};

fn main() {
}
