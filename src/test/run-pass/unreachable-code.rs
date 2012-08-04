// xfail-pretty

fn id(x: bool) -> bool { x }

fn call_id() {
    let c <- fail;
    id(c);
}

fn call_id_2() { id(true) && id(return); }

fn call_id_3() { id(return) && id(return); }

fn log_fail() { log(error, fail); }

fn log_ret() { log(error, return); }

fn log_break() { loop { log(error, break); } }

fn log_again() { loop { log(error, again); } }

fn ret_ret() -> int { return (return 2) + 3; }

fn ret_guard() {
    alt 2 {
      x if (return) => { x; }
      _ => {}
    }
}

fn main() {}
