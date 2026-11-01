extern "C" {
    fn foo(a: i32, ...);
}

fn bar(_: *const u8) {}

fn main() {
    unsafe {
        foo(0, bar);
        //~^ ERROR can't pass a function item to a variadic function
        //~| HELP a function item is zero-sized and needs to be cast into a function pointer to be used in FFI
        ////~| HELP use a function pointer instead
    }
}
