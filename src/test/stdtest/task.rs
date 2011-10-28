use std;
import std::task;
import std::comm;

#[test]
fn test_sleep() { task::sleep(1000000u); }

// FIXME: Leaks on windows
#[test]
#[cfg(target_os = "win32")]
#[ignore]
fn test_unsupervise() { }

#[test]
#[cfg(target_os = "macos")]
#[cfg(target_os = "linux")]
fn test_unsupervise() {
    fn f(&&_i: ()) { task::unsupervise(); fail; }
    task::spawn((), f);
}

#[test]
fn test_lib_spawn() {
    fn foo(&&_i: ()) { log_err "Hello, World!"; }
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
    log_err "received task status message";
    log_err s;
    alt s {
      task::exit(_, task::tr_success.) {/* yay! */ }
      _ { fail "invalid task status received" }
    }
}

// FIXME: Leaks on windows
#[test]
#[cfg(target_os = "win32")]
#[ignore]
fn test_join_chan_fail() { }

#[test]
#[cfg(target_os = "macos")]
#[cfg(target_os = "linux")]
fn test_join_chan_fail() {
    fn failer(&&_i: ()) { task::unsupervise(); fail }

    let p = comm::port();
    task::spawn_notify((), failer, comm::chan(p));
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
    fn winner(&&_i: ()) { }
    let handle = task::spawn_joinable((), winner);
    assert (task::tr_success == task::join(handle));
}

#[test]
#[ignore]
fn spawn_polymorphic() {
    // FIXME #1038: Can't spawn palymorphic functions
    /*fn foo<uniq T>(x: T) { log_err x; }

    task::spawn(true, foo);
    task::spawn(42, foo);*/
}
