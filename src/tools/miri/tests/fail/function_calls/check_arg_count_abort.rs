fn main() {
    extern "C" {
        fn abort(_: i32) -> !;
    }

    unsafe {
        abort(1);
        //~^ ERROR: Undefined Behavior: incorrect number of arguments for `abort`: got 1, expected 0
    }
}
