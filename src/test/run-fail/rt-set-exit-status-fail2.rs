// error-pattern:whatever

class r {
            // Setting the exit status after the runtime has already
            // failed has no effect and the process exits with the
            // runtime's exit code
  drop {
    os::set_exit_status(50);
  }
  new() {}
}

fn main() {
    log(error, "whatever");
    task::spawn {||
      let i = r();
    };
    fail;
}