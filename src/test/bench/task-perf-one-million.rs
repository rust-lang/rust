// Test for concurrent tasks

enum msg {
    ready(comm::chan<msg>),
    start,
    done(int),
}

fn calc(children: uint, parent_ch: comm::chan<msg>) {
    let port = comm::port();
    let chan = comm::chan(port);
    let mut child_chs = ~[];
    let mut sum = 0;

    for iter::repeat (children) {
        do task::spawn {
            calc(0u, chan);
        };
    }

    for iter::repeat (children) {
        alt check comm::recv(port) {
          ready(child_ch) {
            vec::push(child_chs, child_ch);
          }
        }
    }

    comm::send(parent_ch, ready(chan));

    alt check comm::recv(port) {
        start {
          do vec::iter (child_chs) |child_ch| {
              comm::send(child_ch, start);
          }
        }
    }

    for iter::repeat (children) {
        alt check comm::recv(port) {
          done(child_sum) { sum += child_sum; }
        }
    }

    comm::send(parent_ch, done(sum + 1));
}

fn main(args: ~[str]) {
    let args = if os::getenv("RUST_BENCH").is_some() {
        ~["", "100000"]
    } else if args.len() <= 1u {
        ~["", "100"]
    } else {
        args
    };

    let children = uint::from_str(args[1]).get();
    let port = comm::port();
    let chan = comm::chan(port);
    do task::spawn {
        calc(children, chan);
    };
    alt check comm::recv(port) {
      ready(chan) {
        comm::send(chan, start);
      }
    }
    let sum = alt check comm::recv(port) {
      done(sum) { sum }
    };
    #error("How many tasks? %d tasks.", sum);
}
