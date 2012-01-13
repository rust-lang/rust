// error-pattern:whatever

fn main() {
    log(error, "whatever");
    task::spawn {||
        resource r(_i: ()) {
            // Setting the exit status after the runtime has already
            // failed has no effect and the process exits with the
            // runtime's exit code
            sys::set_exit_status(50);
        }
        let i = r(());
    };
    fail;
}