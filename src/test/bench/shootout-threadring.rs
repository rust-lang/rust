// Based on threadring.erlang by Jira Isa
use std;

// FIXME: Need a cleaner way to request the runtime to exit
#[nolink]
native mod libc {
    fn exit(status: ctypes::c_int);
}

const n_threads: int = 503;

fn start(+token: int) {
    import iter::*;

    let p = comm::port();
    let ch = iter::foldl(bind int::range(2, n_threads + 1, _),
                         comm::chan(p)) { |ch, i|
        // FIXME: Some twiddling because we don't have a standard
        // reverse range function yet
        let id = n_threads + 2 - i;
        let {to_child, _} = task::spawn_connected::<int, int> {|p, _ch|
            roundtrip(id, p, ch)
        };
        to_child
    };
    comm::send(ch, token);
    roundtrip(1, p, ch);
}

fn roundtrip(id: int, p: comm::port<int>, ch: comm::chan<int>) {
    while (true) {
        alt comm::recv(p) {
          1 {
            std::io::println(#fmt("%d\n", id));
            libc::exit(0i32);
          }
          token {
            #debug("%d %d", id, token);
            comm::send(ch, token - 1);
          }
        }
    }
}

fn main(args: [str]) {
    let token = if vec::len(args) < 2u {
        1000
    } else {
        int::from_str(args[1])
    };

    start(token);
}