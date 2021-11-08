extern "C" {
    fn foo(a: i32, ...);
}

fn bar(_: *const u8) {}

fn main() {
    unsafe {
        foo(0, bar);
        //~^ ERROR can't pass `fn(*const u8) {bar}` to variadic function
        //~| HELP cast the value to `fn(*const u8)`
    }
}
