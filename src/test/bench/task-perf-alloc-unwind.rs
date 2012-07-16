// xfail-win32

use std;

import std::list::{list, cons, nil};
import std::time::precise_time_s;

fn main() {
    let (repeat, depth) = if os::getenv(~"RUST_BENCH").is_some() {
        (50, 1000)
    } else {
        (10, 10)
    };

    run(repeat, depth);
}

fn run(repeat: int, depth: int) {
    for iter::repeat(repeat as uint) {
        #debug("starting %.4f", precise_time_s());
        do task::try {
            recurse_or_fail(depth, none)
        };
        #debug("stopping %.4f", precise_time_s());
    }
}

type nillist = list<()>;

// Filled with things that have to be unwound
enum st {
    st_({
        box: @nillist,
        unique: ~nillist,
        fn_box: fn@() -> @nillist,
        fn_unique: fn~() -> ~nillist,
        tuple: (@nillist, ~nillist),
        vec: ~[@nillist],
        res: r
    })
}

class r {
  let _l: @nillist;
  new(l: @nillist) { self._l = l; }
  drop {}
}

fn recurse_or_fail(depth: int, st: option<st>) {
    if depth == 0 {
        #debug("unwinding %.4f", precise_time_s());
        fail;
    } else {
        let depth = depth - 1;

        let st = alt st {
          none {
            st_({
                box: @nil,
                unique: ~nil,
                fn_box: fn@() -> @nillist { @nil::<()> },
                fn_unique: fn~() -> ~nillist { ~nil::<()> },
                tuple: (@nil, ~nil),
                vec: ~[@nil],
                res: r(@nil)
            })
          }
          some(st) {
            let fn_box = st.fn_box;
            let fn_unique = st.fn_unique;

            st_({
                box: @cons((), st.box),
                unique: ~cons((), @*st.unique),
                fn_box: fn@() -> @nillist { @cons((), fn_box()) },
                fn_unique: fn~() -> ~nillist { ~cons((), @*fn_unique()) },
                tuple: (@cons((), st.tuple.first()),
                        ~cons((), @*st.tuple.second())),
                vec: st.vec + ~[@cons((), st.vec.last())],
                res: r(@cons((), st.res._l))
            })
          }
        };

        recurse_or_fail(depth, some(st));
    }
}
