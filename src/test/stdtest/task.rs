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
    fn f() { task::unsupervise(); fail; }
    task::spawn {|| f};
}

#[test]
fn test_lib_spawn() {
    fn foo() { #error("Hello, World!"); }
    task::spawn {|| foo};
}

#[test]
fn test_lib_spawn2() {
    fn foo(x: int) { assert (x == 42); }
    task::spawn {|| foo(42);};
}

#[test]
fn test_join_chan() {
    fn winner() { }

    let t = task::spawn_joinable {|| winner();};
    alt task::join(t) {
      task::tr_success. {/* yay! */ }
      _ { fail "invalid task status received" }
    }
}

// FIXME: Leaks on windows
#[test]
#[ignore(cfg(target_os = "win32"))]
fn test_join_chan_fail() {
    fn failer() { task::unsupervise(); fail }

    let t = task::spawn_joinable {|| failer();};
    alt task::join(t) {
      task::tr_failure. {/* yay! */ }
      _ { fail "invalid task status received" }
    }
}

#[test]
fn spawn_polymorphic() {
    fn foo<send T>(x: T) { log(error, x); }
    task::spawn {|| foo(true);}
    task::spawn {|| foo(42);}
}
