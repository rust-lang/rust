fn main() {
    let val = 13usize;
    let addr = &val as *const _ as usize;
    let _ = addr & 13; //~ ERROR access part of a pointer value as raw bytes
}
