// Based on threadring.erlang by Jira Isa
use std;

const n_threads: int = 503;

fn start(+token: int) {
    import iter::*;

    let p = comm::port();
    let mut ch = comm::chan(p);
    int::range(2, n_threads + 1) { |i|
        let id = n_threads + 2 - i;
        let to_child = task::spawn_listener::<int> {|p, copy ch|
            roundtrip(id, p, ch)
        };
        ch = to_child;
    }
    comm::send(ch, token);
    roundtrip(1, p, ch);
}

fn roundtrip(id: int, p: comm::port<int>, ch: comm::chan<int>) {
    while (true) {
        alt comm::recv(p) {
          1 {
            io::println(#fmt("%d\n", id));
            ret;
          }
          token {
            #debug("%d %d", id, token);
            comm::send(ch, token - 1);
            if token <= n_threads {
                ret;
            }
          }
        }
    }
}

fn main(args: [str]) {
    let args = if os::getenv("RUST_BENCH").is_some() {
        ["", "100000"]
    } else if args.len() <= 1u {
        ["", "1000"]
    } else {
        args
    };

    let token = int::from_str(args[1]).get();

    start(token);
}