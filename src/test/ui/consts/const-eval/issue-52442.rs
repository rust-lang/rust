fn main() {
    [();  { &loop { break } as *const _ as usize } ]; //~ ERROR unimplemented expression type
    //~^ ERROR it is undefined behavior to use this value
}
