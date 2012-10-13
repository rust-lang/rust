// xfail-pretty
// xfail-win32

extern mod std;
use std::timer::sleep;
use std::uv;

use pipes::{recv, select};

proto! oneshot (
    waiting:send {
        signal -> !
    }
)

proto! stream (
    stream:send<T:Send> {
        send(T) -> stream<T>
    }
)

fn main() {
    use oneshot::client::*;
    use stream::client::*;

    let iotask = uv::global_loop::get();
    
    let c = pipes::spawn_service(stream::init, |p| { 
        error!("waiting for pipes");
        let stream::send(x, p) = recv(move p);
        error!("got pipes");
        let (left, right) : (oneshot::server::waiting,
                             oneshot::server::waiting)
            = move x;
        error!("selecting");
        let (i, _, _) = select(~[move left, move right]);
        error!("selected");
        assert i == 0;

        error!("waiting for pipes");
        let stream::send(x, _) = recv(move p);
        error!("got pipes");
        let (left, right) : (oneshot::server::waiting,
                             oneshot::server::waiting)
            = move x;
        error!("selecting");
        let (i, m, _) = select(~[move left, move right]);
        error!("selected %?", i);
        if m.is_some() {
            assert i == 1;
        }
    });

    let (c1, p1) = oneshot::init();
    let (_c2, p2) = oneshot::init();

    let c = send(move c, (move p1, move p2));
    
    sleep(iotask, 100);

    signal(move c1);

    let (_c1, p1) = oneshot::init();
    let (c2, p2) = oneshot::init();

    send(move c, (move p1, move p2));

    sleep(iotask, 100);

    signal(move c2);

    test_select2();
}

fn test_select2() {
    let (ac, ap) = stream::init();
    let (bc, bp) = stream::init();

    stream::client::send(move ac, 42);

    match pipes::select2(move ap, move bp) {
      either::Left(*) => { }
      either::Right(*) => { fail }
    }

    stream::client::send(move bc, ~"abc");

    error!("done with first select2");

    let (ac, ap) = stream::init();
    let (bc, bp) = stream::init();

    stream::client::send(move bc, ~"abc");

    match pipes::select2(move ap, move bp) {
      either::Left(*) => { fail }
      either::Right(*) => { }
    }

    stream::client::send(move ac, 42);
}
