fn main() {
    static foo: dyn Fn() -> u32 = || -> u32 {
        //~^ ERROR the size for values of type
        0
    };
}
