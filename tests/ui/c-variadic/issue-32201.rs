extern "C" {
    fn foo(a: i32, ...);
}

fn bar(_: *const u8) {}

fn main() {
    unsafe {
        foo(0, bar);
        //~^ ERROR can't pass `{fn item bar: fn(*const u8)}` to variadic function
        //~| HELP a function item is zero-sized and needs to be casted into a function pointer to be used in FFI
        //~| HELP cast the value into a function pointer
        //~| HELP cast the value to `fn(*const u8)
    }
}
