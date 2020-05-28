fn main() {
    [();  { &loop { break } as *const _ as usize } ];
    //~^ ERROR `loop` is not allowed in a `const`
    //~| ERROR casting pointers to integers in constants is unstable
    //~| ERROR evaluation of constant value failed
}
