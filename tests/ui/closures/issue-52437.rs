fn main() {
    [(); &(&'static: loop { |x| {}; }) as *const _ as usize]
    //~^ ERROR: labels cannot use keyword names
    //~| ERROR: type annotations needed
}
