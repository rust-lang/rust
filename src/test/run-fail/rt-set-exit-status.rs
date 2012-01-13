// error-pattern:whatever

fn main() {
    log(error, "whatever");
    // 101 is the code the runtime uses on task failure and the value
    // compiletest expects run-fail tests to return.
    sys::set_exit_status(101);
}