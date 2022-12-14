fn main() {
    const X: u32 = 1;
    const Y: usize = unsafe { &X as *const u32 as usize }; //~ ERROR pointers cannot be cast to integers
    println!("{}", Y);
}
