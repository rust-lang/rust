fn main() {
    extern "C" {
        fn abort(_: i32) -> !;
    }

    unsafe {
        abort(1);
        //~^ ERROR: takes 0 arguments, but 1 argument was given
    }
}
