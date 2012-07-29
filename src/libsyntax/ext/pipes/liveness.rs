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

import dvec::extensions;

import std::bitv::{bitv};

import proto::methods;
import ast_builder::empty_span;

fn analyze(proto: protocol, _cx: ext_ctxt) {
    #debug("initializing colive analysis");
    let num_states = proto.num_states();
    let colive = do (copy proto.states).map_to_vec |state| {
        let bv = ~bitv(num_states, false);
        for state.reachable |s| {
            bv.set(s.id, true);
        }
        bv
    };

    let mut i = 0;
    let mut changed = true;
    while changed {
        changed = false;
        #debug("colive iteration %?", i);
        for colive.eachi |i, this_colive| {
            let this = proto.get_state_by_id(i);
            for this_colive.ones |j| {
                let next = proto.get_state_by_id(j);
                if this.dir == next.dir {
                    changed = changed || this_colive.union(colive[j]);
                }
            }
        }
        i += 1;
    }

    #debug("colive analysis complete");

    // Determine if we're bounded
    let mut self_live = ~[];
    for colive.eachi |i, bv| {
        if bv.get(i) {
            vec::push(self_live, proto.get_state_by_id(i))
        }
    }

    if self_live.len() > 0 {
        let states = str::connect(self_live.map(|s| *s.name), ~" ");

        #debug("protocol %s is unbounded due to loops involving: %s",
               *proto.name, states);

        // Someday this will be configurable with a warning
        //cx.span_warn(empty_span(),
        //              #fmt("protocol %s is unbounded due to loops \
        //                    involving these states: %s",
        //                   *proto.name,
        //                   states));

        proto.bounded = some(false);
    }
    else {
        #debug("protocol %s is bounded. yay!", *proto.name);
        proto.bounded = some(true);
    }
}