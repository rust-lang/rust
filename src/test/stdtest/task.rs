use std;
import std::task;

#[test]
#[ignore]
fn test_sleep() { task::sleep(1000000u); }

#[test]
fn test_unsupervise() {
    fn f() {
        task::unsupervise();
        fail;
    }
    spawn f();
}

#[test]
fn test_join() {
    fn winner() {
    }

    auto wintask = spawn winner();

    assert task::join(wintask) == task::tr_success;

    fn failer() {
        task::unsupervise();
        fail;
    }

    auto failtask = spawn failer();

    assert task::join(failtask) == task::tr_failure;
}

#[test]
fn test_send_recv() {
    auto p = port[int]();
    auto c = chan(p);
    task::send(c, 10);
    assert task::recv(p) == 10;
}

#[test]
fn test_worker() {
    task::worker(fn(port[int] p) {
        auto x; p |> x;
        assert x == 10;
    }).chan <| 10;

    task::worker(fn(port[rec(int x, int y)] p) {
        auto x; p |> x;
        assert x.y == 20;
    }).chan <| rec(x = 10, y = 20);

    task::worker(fn(port[rec(int x, int y, int z)] p) {
        auto x; p |> x;
        assert x.z == 30;
    }).chan <| rec(x = 10, y = 20, z = 30);

    task::worker(fn(port[rec(int a, int b, int c, int d)] p) {
        auto x; p |> x;
        assert x.d == 40;
    }).chan <| rec(a = 10, b = 20, c = 30, d = 40);
}
