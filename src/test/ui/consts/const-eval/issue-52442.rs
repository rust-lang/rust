fn main() {
    [();  { &loop { break } as *const _ as usize } ]; //~ ERROR unimplemented expression type
}
