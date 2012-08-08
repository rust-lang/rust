/// Correctness for protocols

/*

This section of code makes sure the protocol is likely to generate
correct code. The correctness criteria include:

  * No protocols transition to states that don't exist.
  * Messages step to states with the right number of type parameters.

In addition, this serves as a lint pass. Lint warns for the following
things.

  * States with no messages, it's better to step to !.

It would also be nice to warn about unreachable states, but the
visitor infrastructure for protocols doesn't currently work well for
that.

*/

import ext::base::ext_ctxt;

import ast::{ident};

import proto::{state, protocol, next_state};
import ast_builder::empty_span;

impl ext_ctxt: proto::visitor<(), (), ()>  {
    fn visit_proto(_proto: protocol,
                   _states: &[()]) { }

    fn visit_state(state: state, _m: &[()]) {
        if state.messages.len() == 0 {
            self.span_warn(
                state.span, // use a real span!
                fmt!{"state %s contains no messages, \
                      consider stepping to a terminal state instead",
                     *state.name})
        }
    }

    fn visit_message(name: ident, _span: span, _tys: &[@ast::ty],
                     this: state, next: next_state) {
        match next {
          some({state: next, tys: next_tys}) => {
            let proto = this.proto;
            if !proto.has_state(next) {
                // This should be a span fatal, but then we need to
                // track span information.
                self.span_err(
                    proto.get_state(next).span,
                    fmt!{"message %s steps to undefined state, %s",
                         *name, *next});
            }
            else {
                let next = proto.get_state(next);

                if next.ty_params.len() != next_tys.len() {
                    self.span_err(
                        next.span, // use a real span
                        fmt!{"message %s target (%s) \
                              needs %u type parameters, but got %u",
                             *name, *next.name,
                             next.ty_params.len(),
                             next_tys.len()});
                }
            }
          }
          none => ()
        }
    }
}