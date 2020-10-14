fn main() {
    [(); &(&'static: loop { |x| {}; }) as *const _ as usize]
    //~^ ERROR: invalid label name `'static`
    //~| ERROR mismatched types
}
