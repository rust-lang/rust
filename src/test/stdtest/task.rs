use std;
import std::task;
import std::comm;

#[test]
fn test_sleep() { task::sleep(1000000u); }

#[test]
fn test_unsupervise() {
    fn f() { task::unsupervise(); fail; }
    let foo = f;
    task::_spawn(foo);
}

#[test]
#[ignore]
fn test_join() {
    fn winner() { }

    let wintask = task::_spawn(bind winner());

    assert (task::join_id(wintask) == task::tr_success);

    fn failer() { task::unsupervise(); fail; }

    let failtask = task::_spawn(bind failer());

    assert (task::join_id(failtask) == task::tr_failure);
}

#[test]
fn test_lib_spawn() {
    fn foo() { log_err "Hello, World!"; }
    let f = foo;
    task::_spawn(f);
}

#[test]
fn test_lib_spawn2() {
    fn foo(x: int) { assert (x == 42); }
    task::_spawn(bind foo(42));
}

#[test]
fn test_join_chan() {
    fn winner() { }

    let p = comm::mk_port::<task::task_notification>();
    let f = winner;
    task::spawn_notify(f, p.mk_chan());
    let s = p.recv();
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

    let p = comm::mk_port::<task::task_notification>();
    let f = failer;
    task::spawn_notify(f, p.mk_chan());
    let s = p.recv();
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
