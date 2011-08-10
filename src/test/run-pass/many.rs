// -*- rust -*-

use std;
import std::task;
import std::comm;

fn sub(parent: comm::_chan<int>, id: int) {
  if (id == 0) {
      comm::send(parent, 0);
  } else {
      let p = comm::mk_port();
      let child = task::_spawn(bind sub(p.mk_chan(), id-1));
      let y = p.recv();
      comm::send(parent, y + 1);
  }
}

fn main() {
  let p = comm::mk_port();
  let child = task::_spawn(bind sub(p.mk_chan(), 200));
  let y = p.recv();
  log "transmission complete";
  log y;
  assert (y == 200);
}
