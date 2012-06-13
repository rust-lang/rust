#[link(name = "zmq",
       vers = "0.2",
       uuid = "54cc1bc9-02b8-447c-a227-75ebc923bc29")];
#[crate_type = "lib"];

use std;

export context;

resource arc_destruct<T: const>(_data: int) { }

fn arc<T: const>(_data: T) -> arc_destruct<T> {
    arc_destruct(0)
}

fn init() -> arc_destruct<context_res> unsafe {
    arc(context_res())
}

class context_res {
    let ctx : int;

    new() { self.ctx = 0; }

    drop { }
}

type context = arc_destruct<context_res>;

impl context for context {
    fn socket() { }
}
