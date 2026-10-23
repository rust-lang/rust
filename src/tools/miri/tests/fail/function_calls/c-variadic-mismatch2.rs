//! Caller is variadic, callee not.
mod module {
    #[unsafe(no_mangle)]
    extern "C" fn weird() {}
}

unsafe extern "C" {
    fn weird(_: ...);
}

fn main() {
    unsafe { weird(1_i32) };
    //~^ ERROR: Undefined Behavior: calling a function where the caller and callee disagree on whether the function is C-variadic
}
