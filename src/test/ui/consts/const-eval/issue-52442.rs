fn main() {
    [();  { &loop { break } as *const _ as usize } ];
    //~^ ERROR casting pointers to integers in constants is unstable
    //~| ERROR `loop` is not allowed in a `const`
    //~| ERROR evaluation of constant value failed
}
