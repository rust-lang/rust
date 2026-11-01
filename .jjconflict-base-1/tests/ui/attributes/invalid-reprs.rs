fn main() {
    let y = #[repr(uwu(4))]
    //~^ ERROR attributes on expressions are experimental
    //~| ERROR malformed `repr` attribute input
    (&id(5)); //~ ERROR: cannot find function `id` in this scope
}
