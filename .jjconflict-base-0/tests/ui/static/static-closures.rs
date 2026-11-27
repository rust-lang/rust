fn main() {
    static || {};
    //~^ ERROR closures cannot be static
    //~| ERROR coroutine syntax is experimental
}
