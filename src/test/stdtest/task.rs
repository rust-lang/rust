use std;
import std::task;

#[test]
fn test_sleep() { task::sleep(1000000u); }

#[test]
fn test_unsupervise() {
    fn f() { task::unsupervise(); fail; }
    spawn f();
}

#[test]
fn test_join() {
    fn winner() { }

    let wintask = spawn winner();

    assert (task::join(wintask) == task::tr_success);

    fn failer() { task::unsupervise(); fail; }

    let failtask = spawn failer();

    assert (task::join(failtask) == task::tr_failure);
}

#[test]
fn test_send_recv() {
    let p = port[int]();
    let c = chan(p);
    task::send(c, 10);
    assert (task::recv(p) == 10);
}

