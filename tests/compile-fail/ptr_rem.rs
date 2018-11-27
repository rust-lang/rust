fn main() {
    let val = 13usize;
    let addr = &val as *const _ as usize;
    let _val = addr % 16; //~ ERROR access part of a pointer value as raw bytes
}
