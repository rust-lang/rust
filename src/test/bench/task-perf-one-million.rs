// xfail-test FIXME: Can't run under valgrind - too much RAM
// FIXME: This doesn't spawn close to a million tasks yet

tag msg {
    ready(comm::chan<msg>);
    start;
    done(int);
}

fn calc(&&args: (int, comm::chan<msg>)) {
    let (depth, parent_ch) = args;
    let port = comm::port();
    let children = depth > 0 ? 20u : 0u;
    let child_chs = [];
    let sum = 0;

    repeat (children) {||
        task::spawn((depth - 1, comm::chan(port)), calc);
    }

    repeat (children) {||
        alt comm::recv(port) {
          ready(child_ch) {
            child_chs += [child_ch];
          }
        }
    }

    comm::send(parent_ch, ready(comm::chan(port)));

    alt comm::recv(port) {
        start. {
          vec::iter (child_chs) { |child_ch|
              comm::send(child_ch, start);
          }
        }
    }

    repeat (children) {||
        alt comm::recv(port) {
          done(child_sum) { sum += child_sum; }
        }
    }

    comm::send(parent_ch, done(sum + 1));
}

fn main() {
    let port = comm::port();
    task::spawn((3, comm::chan(port)), calc);
    alt comm::recv(port) {
      ready(chan) {
        comm::send(chan, start);
      }
    }
    let sum = alt comm::recv(port) {
      done(sum) { sum }
    };
    log #fmt("How many tasks? That's right, %d tasks.", sum);
}