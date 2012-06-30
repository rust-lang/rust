use std;
import std::timer::sleep;
import std::uv;

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

fn main() {
    import oneshot::client::*;
    import oneshot::server::recv;

    #macro[
        [#recv[chan],
         chan.recv()(chan)]
    ];

    let c = pipes::spawn_service(oneshot::init) {|p|
        #recv(p);
    };

    let iotask = uv::global_loop::get();
    sleep(iotask, 5000);
    
    signal(c);
}