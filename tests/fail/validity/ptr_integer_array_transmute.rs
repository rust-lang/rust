fn main() {
    let r = &mut 42;
    let _i: [usize; 1] = unsafe { std::mem::transmute(r) }; //~ ERROR: encountered a pointer, but expected plain (non-pointer) bytes
}
