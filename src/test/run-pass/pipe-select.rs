use std;
import std::timer::sleep;
import std::uv;

import pipes::{recv, select};

// Compiled by pipec
mod oneshot {
    fn init() -> (client::waiting, server::waiting) { pipes::entangle() }
    enum waiting { signal(server::signaled), }
    enum signaled { }
    mod client {
        fn signal(-pipe: waiting) -> signaled {
            let (c, s) = pipes::entangle();
            let message = oneshot::signal(s);
            pipes::send(pipe, message);
            c
        }
        type waiting = pipes::send_packet<oneshot::waiting>;
        type signaled = pipes::send_packet<oneshot::signaled>;
    }
    mod server {
        impl recv for waiting {
            fn recv() -> extern fn(-waiting) -> oneshot::waiting {
                fn recv(-pipe: waiting) -> oneshot::waiting {
                    option::unwrap(pipes::recv(pipe))
                }
                recv
            }
        }
        type waiting = pipes::recv_packet<oneshot::waiting>;
        impl recv for signaled {
            fn recv() -> extern fn(-signaled) -> oneshot::signaled {
                fn recv(-pipe: signaled) -> oneshot::signaled {
                    option::unwrap(pipes::recv(pipe))
                }
                recv
            }
        }
        type signaled = pipes::recv_packet<oneshot::signaled>;
    }
}

mod stream {
    fn init<T: send>() -> (client::stream<T>, server::stream<T>) {
        pipes::entangle()
    }
    enum stream<T: send> { send(T, server::stream<T>), }
    mod client {
        fn send<T: send>(+pipe: stream<T>, +x_0: T) -> stream<T> {
            {
                let (c, s) = pipes::entangle();
                let message = stream::send(x_0, s);
                pipes::send(pipe, message);
                c
            }
        }
        type stream<T: send> = pipes::send_packet<stream::stream<T>>;
    }
    mod server {
        impl recv<T: send> for stream<T> {
            fn recv() -> extern fn(+stream<T>) -> stream::stream<T> {
                fn recv<T: send>(+pipe: stream<T>) -> stream::stream<T> {
                    option::unwrap(pipes::recv(pipe))
                }
                recv
            }
        }
        type stream<T: send> = pipes::recv_packet<stream::stream<T>>;
    }
}

fn main() {
    import oneshot::client::*;
    import stream::client::*;

    let iotask = uv::global_loop::get();
    
    #macro[
        [#recv[chan],
         chan.recv()(chan)]
    ];

    let c = pipes::spawn_service(stream::init, |p| { 
        #error("waiting for pipes");
        let stream::send(x, p) = option::unwrap(recv(p));
        #error("got pipes");
        let (left, right) : (oneshot::server::waiting,
                             oneshot::server::waiting)
            = x;
        #error("selecting");
        let (i, _, _) = select(~[left, right]);
        #error("selected");
        assert i == 0;

        #error("waiting for pipes");
        let stream::send(x, _) = option::unwrap(recv(p));
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
}