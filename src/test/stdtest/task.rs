use std;
import std::task;

#[test]
fn test_sleep() { task::sleep(1000000u); }

#[test]
fn test_unsupervise() {
    fn f() { task::unsupervise(); fail; }
    task::_spawn(bind f());
}

#[test]
fn test_join() {
    fn winner() { }

    let wintask = task::_spawn(bind winner());

    assert (task::join_id(wintask) == task::tr_success);

    fn failer() { task::unsupervise(); fail; }

    let failtask = task::_spawn(bind failer());

    assert (task::join_id(failtask) == task::tr_failure);
}

#[test]
fn test_send_recv() {
    let p = port[int]();
    let c = chan(p);
    task::send(c, 10);
    assert (task::recv(p) == 10);
}

#[test]
fn test_lib_spawn() {
    fn foo() { log_err "Hello, World!"; }
    task::_spawn(foo);
}

#[test]
fn test_lib_spawn2() {
    fn foo(x : int) { assert(x == 42); }
    task::_spawn(bind foo(42));
}
