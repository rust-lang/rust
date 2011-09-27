// xfail-pretty

fn id(x: bool) -> bool { x }

fn call_id() {
    let c <- fail;
    id(c);
}

fn call_id_2() { id(true) && id(ret); }

fn call_id_3() { id(ret) && id(ret); }

fn call_id_4() { while id(break) { } }

fn bind_id_1() { bind id(fail); }

fn bind_id_2() { bind id(ret); }

iter put_break() -> int {
    while true { put break; }
}

fn fail_fail() { fail fail; }

fn log_fail() { log_err fail; }

fn log_ret() { log_err ret; }

fn log_break() { while true { log_err break; } }

fn log_cont() { do { log_err cont; } while false }

fn ret_ret() -> int { ret (ret 2) + 3; }

fn ret_guard() {
    alt 2 {
      x when (ret) { x; }
    }
}

fn rec_ret() { let _r = {c: ret}; }

fn vec_ret() { let _v = [1, 2, ret, 4]; }

fn fail_then_concat() {
    let x = [], y = [3];
    fail;
    x += y;
    "good" + "bye";
}

fn main() {
  // Call the functions that don't fail.
  rec_ret();
  vec_ret();
  ret_ret();
  log_ret();
  call_id_2();
  call_id_3();
  call_id_4();
  bind_id_2();
  ret_guard();
}
