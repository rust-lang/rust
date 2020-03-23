fn main() {
    [();  { &loop { break } as *const _ as usize } ];
    //~^ ERROR `loop` is not allowed in a `const`
}
