// xfail-pretty

use std;
import std::timer::sleep;
import std::uv;

import pipes::{recv, select};

proto! oneshot {
    waiting:send {
        signal -> signaled
    }

    signaled:send { }
}

proto! stream {
    stream:send<T:send> {
        send(T) -> stream<T>
    }
}

fn main() {
    import oneshot::client::*;
    import stream::client::*;

    let iotask = uv::global_loop::get();
    
    let c = pipes::spawn_service(stream::init, |p| { 
        #error("waiting for pipes");
        let stream::send(x, p) = recv(p);
        #error("got pipes");
        let (left, right) : (oneshot::server::waiting,
                             oneshot::server::waiting)
            = x;
        #error("selecting");
        let (i, _, _) = select(~[left, right]);
        #error("selected");
        assert i == 0;

        #error("waiting for pipes");
        let stream::send(x, _) = recv(p);
        #error("got pipes");
        let (left, right) : (oneshot::server::waiting,
                             oneshot::server::waiting)
            = x;
        #error("selecting");
        let (i, _, _) = select(~[left, right]);
        #error("selected");
        assert i == 1;
    });

    let (c1, p1) = oneshot::init();
    let (c2, p2) = oneshot::init();

    let c = send(c, (p1, p2));
    
    sleep(iotask, 1000);

    signal(c1);

    let (c1, p1) = oneshot::init();
    let (c2, p2) = oneshot::init();

    send(c, (p1, p2));

    sleep(iotask, 1000);

    signal(c2);

    test_select2();
}

fn test_select2() {
    let (ac, ap) = stream::init();
    let (bc, bp) = stream::init();

    stream::client::send(ac, 42);

    alt pipes::select2(ap, bp) {
      either::left(*) { }
      either::right(*) { fail }
    }

    stream::client::send(bc, "abc");

    #error("done with first select2");

    let (ac, ap) = stream::init();
    let (bc, bp) = stream::init();

    stream::client::send(bc, "abc");

    alt pipes::select2(ap, bp) {
      either::left(*) { fail }
      either::right(*) { }
    }

    stream::client::send(ac, 42);
}
