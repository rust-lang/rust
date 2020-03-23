fn main() {
    [(); &(&'static: loop { |x| {}; }) as *const _ as usize]
    //~^ ERROR: invalid label name `'static`
    //~| ERROR: `loop` is not allowed in a `const`
    //~| ERROR: type annotations needed
    //~| ERROR mismatched types
}
