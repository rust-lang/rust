import core::*;

use std;
import task;
import comm;

#[test]
fn test_sleep() { task::sleep(1000000u); }

// FIXME: Leaks on windows
#[test]
#[ignore(cfg(target_os = "win32"))]
fn test_unsupervise() {
    fn f(&&_i: ()) { task::unsupervise(); fail; }
    task::spawn((), f);
}

#[test]
fn test_lib_spawn() {
    fn foo(&&_i: ()) { #error("Hello, World!"); }
    task::spawn((), foo);
}

#[test]
fn test_lib_spawn2() {
    fn foo(&&x: int) { assert (x == 42); }
    task::spawn(42, foo);
}

#[test]
fn test_join_chan() {
    fn winner(&&_i: ()) { }

    let p = comm::port();
    task::spawn_notify((), winner, comm::chan(p));
    let s = comm::recv(p);
    #error("received task status message");
    log(error, s);
    alt s {
      task::exit(_, task::tr_success.) {/* yay! */ }
      _ { fail "invalid task status received" }
    }
}

// FIXME: Leaks on windows
#[test]
#[ignore(cfg(target_os = "win32"))]
fn test_join_chan_fail() {
    fn failer(&&_i: ()) { task::unsupervise(); fail }

    let p = comm::port();
    task::spawn_notify((), failer, comm::chan(p));
    let s = comm::recv(p);
    #error("received task status message");
    log(error, s);
    alt s {
      task::exit(_, task::tr_failure.) {/* yay! */ }
      _ { fail "invalid task status received" }
    }
}

#[test]
fn test_join_convenient() {
    fn winner(&&_i: ()) { }
    let handle = task::spawn_joinable((), winner);
    assert (task::tr_success == task::join(handle));
}

#[test]
#[ignore]
fn spawn_polymorphic() {
    // FIXME #1038: Can't spawn palymorphic functions
    /*fn foo<send T>(x: T) { log(error, x); }

    task::spawn(true, foo);
    task::spawn(42, foo);*/
}
