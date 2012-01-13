// error-pattern:whatever

fn main() {
    log(error, "whatever");
    // Setting the exit status only works when the scheduler terminates
    // normally. In this case we're going to fail, so instead of of
    // returning 50 the process will return the typical rt failure code.
    sys::set_exit_status(50);
    fail;
}