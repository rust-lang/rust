fn main() {
    extern "C" {
        fn abort(_: i32) -> !;
    }

    unsafe {
        abort(1);
        //~^ ERROR: expected 0 arguments, found 1 arguments
    }
}
