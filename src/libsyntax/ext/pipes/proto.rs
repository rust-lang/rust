import to_str::to_str;
import dvec::{dvec, extensions};

import ast::{ident};

import ast_builder::{path, methods, ast_builder, append_types};

enum direction {
    send, recv
}

impl of to_str for direction {
    fn to_str() -> ~str {
        alt self {
          send { ~"send" }
          recv { ~"recv" }
        }
    }
}

impl methods for direction {
    fn reverse() -> direction {
        alt self {
          send { recv }
          recv { send }
        }
    }
}

type next_state = option<{state: ident, tys: ~[@ast::ty]}>;

enum message {
    // name, data, current state, next state
    message(ident, ~[@ast::ty], state, next_state)
}

impl methods for message {
    fn name() -> ident {
        alt self {
          message(id, _, _, _) {
            id
          }
        }
    }

    /// Return the type parameters actually used by this message
    fn get_params() -> ~[ast::ty_param] {
        alt self {
          message(_, _, this, _) {
            this.ty_params
          }
        }
    }
}

enum state {
    state_(@{
        id: uint,
        name: ident,
        dir: direction,
        ty_params: ~[ast::ty_param],
        messages: dvec<message>,
        proto: protocol,
    }),
}

impl methods for state {
    fn add_message(name: ident, +data: ~[@ast::ty], next: next_state) {
        self.messages.push(message(name, data, self,
                                   next));
    }

    fn filename() -> ~str {
        (*self).proto.filename()
    }

    fn data_name() -> ident {
        self.name
    }

    fn to_ty(cx: ext_ctxt) -> @ast::ty {
        cx.ty_path_ast_builder
            (path(self.name).add_tys(cx.ty_vars(self.ty_params)))
    }

    /// Iterate over the states that can be reached in one message
    /// from this state.
    fn reachable(f: fn(state) -> bool) {
        for self.messages.each |m| {
            alt m {
              message(_, _, _, some({state: id, _})) {
                let state = self.proto.get_state(id);
                if !f(state) { break }
              }
              _ { }
            }
        }
    }
}

enum protocol {
    protocol_(@{
        name: ident,
        states: dvec<state>,
    }),
}

fn protocol(name: ident) -> protocol {
    protocol_(@{name: name, states: dvec()})
}

impl methods for protocol {
    fn add_state(name: ident, dir: direction) -> state {
        self.add_state_poly(name, dir, ~[])
    }

    /// Get or create a state.
    fn get_state(name: ident) -> state {
        self.states.find(|i| i.name == name).get()
    }

    fn get_state_by_id(id: uint) -> state { self.states[id] }

    fn has_state(name: ident) -> bool {
        self.states.find(|i| i.name == name) != none
    }

    fn add_state_poly(name: ident, dir: direction,
                      +ty_params: ~[ast::ty_param]) -> state {
        let messages = dvec();

        let state = state_(@{
            id: self.states.len(),
            name: name,
            dir: dir,
            ty_params: ty_params,
            messages: messages,
            proto: self
        });

        self.states.push(state);
        state
    }

    fn filename() -> ~str {
        ~"proto://" + *self.name
    }

    fn num_states() -> uint { self.states.len() }
}

trait visitor<Tproto, Tstate, Tmessage> {
    fn visit_proto(proto: protocol, st: &[Tstate]) -> Tproto;
    fn visit_state(state: state, m: &[Tmessage]) -> Tstate;
    fn visit_message(name: ident, tys: &[@ast::ty],
                     this: state, next: next_state) -> Tmessage;
}

fn visit<Tproto, Tstate, Tmessage, V: visitor<Tproto, Tstate, Tmessage>>(
    proto: protocol, visitor: V) -> Tproto {

    // the copy keywords prevent recursive use of dvec
    let states = do (copy proto.states).map_to_vec |s| {
        let messages = do (copy s.messages).map_to_vec |m| {
            let message(name, tys, this, next) = m;
            visitor.visit_message(name, tys, this, next)
        };
        visitor.visit_state(s, messages)
    };
    visitor.visit_proto(proto, states)
}
