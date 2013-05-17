// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

/*

Liveness analysis for protocols. This is useful for a lot of possible
optimizations.

This analysis computes the "co-live" relationship between
states. Co-live is defined inductively as follows.

1. u is co-live with v if u can transition to v in one message.

2. u is co-live with v if there exists a w such that u and w are
co-live, w and v are co-live, and u and w have the same direction.

This relationship approximates when it is safe to store two states in
the same memory location. If there is no u such u is co-live with
itself, then the protocol is bounded.

(These assertions could use proofs)

In addition, this analysis does reachability, to warn when we have
useless states.

The algorithm is a fixpoint computation. For each state, we initialize
a bitvector containing whether it is co-live with each other state. At
first we use rule (1) above to set each vector. Then we iterate
updating the states using rule (2) until there are no changes.

*/

use ext::base::ExtCtxt;
use ext::pipes::proto::{protocol_};

use std::bitv::Bitv;

pub fn analyze(proto: @mut protocol_, _cx: @ExtCtxt) {
    debug!("initializing colive analysis");
    let num_states = proto.num_states();
    let mut colive = do (copy proto.states).map_to_vec |state| {
        let mut bv = ~Bitv::new(num_states, false);
        for state.reachable |s| {
            bv.set(s.id, true);
        }
        bv
    };

    let mut i = 0;
    let mut changed = true;
    while changed {
        changed = false;
        debug!("colive iteration %?", i);
        let mut new_colive = ~[];
        for colive.eachi |i, this_colive| {
            let mut result = this_colive.clone();
            let this = proto.get_state_by_id(i);
            for this_colive.ones |j| {
                let next = proto.get_state_by_id(j);
                if this.dir == next.dir {
                    changed = result.union(colive[j]) || changed;
                }
            }
            new_colive.push(result)
        }
        colive = new_colive;
        i += 1;
    }

    debug!("colive analysis complete");

    // Determine if we're bounded
    let mut self_live = ~[];
    for colive.eachi |i, bv| {
        if bv.get(i) {
            self_live.push(proto.get_state_by_id(i))
        }
    }

    if self_live.len() > 0 {
        let states = str::connect(self_live.map(|s| copy s.name), " ");

        debug!("protocol %s is unbounded due to loops involving: %s",
               copy proto.name, states);

        // Someday this will be configurable with a warning
        //cx.span_warn(empty_span(),
        //              fmt!("protocol %s is unbounded due to loops \
        //                    involving these states: %s",
        //                   *proto.name,
        //                   states));

        proto.bounded = Some(false);
    }
    else {
        debug!("protocol %s is bounded. yay!", copy proto.name);
        proto.bounded = Some(true);
    }
}
