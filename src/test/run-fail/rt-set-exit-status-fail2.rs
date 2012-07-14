// error-pattern:whatever

class r {
  let x:int;
            // Setting the exit status after the runtime has already
            // failed has no effect and the process exits with the
            // runtime's exit code
  drop {
    os::set_exit_status(50);
  }
  new(x:int) {self.x = x;}
}

fn main() {
    log(error, ~"whatever");
    do task::spawn {
      let i = r(5);
    };
    fail;
}