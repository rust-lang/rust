use std;
import std::task;
import std::comm;

#[test]
fn test_sleep() { task::sleep(1000000u); }

#[test]
fn test_unsupervise() {
    fn f() { task::unsupervise(); fail; }
    let foo = f;
    task::spawn(foo);
}

#[test]
fn test_lib_spawn() {
    fn foo() { log_err "Hello, World!"; }
    let f = foo;
    task::spawn(f);
}

#[test]
fn test_lib_spawn2() {
    fn foo(x: int) { assert (x == 42); }
    task::spawn(bind foo(42));
}

#[test]
fn test_join_chan() {
    fn winner() { }

    let p = comm::port();
    let f = winner;
    task::spawn_notify(f, comm::chan(p));
    let s = comm::recv(p);
    log_err "received task status message";
    log_err s;
    alt s {
      task::exit(_, task::tr_success.) {/* yay! */ }
      _ { fail "invalid task status received" }
    }
}

#[test]
fn test_join_chan_fail() {
    fn failer() { task::unsupervise(); fail }

    let p = comm::port();
    let f = failer;
    task::spawn_notify(f, comm::chan(p));
    let s = comm::recv(p);
    log_err "received task status message";
    log_err s;
    alt s {
      task::exit(_, task::tr_failure.) {/* yay! */ }
      _ { fail "invalid task status received" }
    }
}

#[test]
fn test_join_convenient() {
    fn winner() { }
    let f = winner;
    let handle = task::spawn_joinable(f);
    assert (task::tr_success == task::join(handle));
}

#[test]
fn spawn_polymorphic() {
    fn foo<~T>(x: -T) { log_err x; }

    let fb = bind foo(true);

    task::spawn(fb);
    task::spawn(bind foo(42));
}
