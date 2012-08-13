import to_str::ToStr;
import dvec::dvec;

import ast::{ident};

import ast_builder::{path, append_types};

enum direction {
    send, recv
}

impl direction: ToStr {
    fn to_str() -> ~str {
        match self {
          send => ~"send",
          recv => ~"recv"
        }
    }
}

impl direction {
    fn reverse() -> direction {
        match self {
          send => recv,
          recv => send
        }
    }
}

type next_state = option<{state: ident, tys: ~[@ast::ty]}>;

enum message {
    // name, span, data, current state, next state
    message(ident, span, ~[@ast::ty], state, next_state)
}

impl message {
    fn name() -> ident {
        match self {
          message(id, _, _, _, _) => id
        }
    }

    fn span() -> span {
        match self {
          message(_, span, _, _, _) => span
        }
    }

    /// Return the type parameters actually used by this message
    fn get_params() -> ~[ast::ty_param] {
        match self {
          message(_, _, _, this, _) => this.ty_params
        }
    }
}

enum state {
    state_(@{
        id: uint,
        name: ident,
        span: span,
        dir: direction,
        ty_params: ~[ast::ty_param],
        messages: dvec<message>,
        proto: protocol,
    }),
}

impl state {
    fn add_message(name: ident, span: span,
                   +data: ~[@ast::ty], next: next_state) {
        self.messages.push(message(name, span, data, self,
                                   next));
    }

    fn filename() -> ~str {
        (*self).proto.filename()
    }

    fn data_name() -> ident {
        self.name
    }

    /// Returns the type that is used for the messages.
    fn to_ty(cx: ext_ctxt) -> @ast::ty {
        cx.ty_path_ast_builder
            (path(self.name, self.span).add_tys(cx.ty_vars(self.ty_params)))
    }

    /// Iterate over the states that can be reached in one message
    /// from this state.
    fn reachable(f: fn(state) -> bool) {
        for self.messages.each |m| {
            match m {
              message(_, _, _, _, some({state: id, _})) => {
                let state = self.proto.get_state(id);
                if !f(state) { break }
              }
              _ => ()
            }
        }
    }
}

type protocol = @protocol_;

fn protocol(name: ident, +span: span) -> protocol {
    @protocol_(name, span)
}

class protocol_ {
    let name: ident;
    let span: span;
    let states: dvec<state>;

    let mut bounded: option<bool>;

    new(name: ident, span: span) {
        self.name = name;
        self.span = span;
        self.states = dvec();
        self.bounded = none;
    }

    /// Get a state.
    fn get_state(name: ident) -> state {
        self.states.find(|i| i.name == name).get()
    }

    fn get_state_by_id(id: uint) -> state { self.states[id] }

    fn has_state(name: ident) -> bool {
        self.states.find(|i| i.name == name) != none
    }

    fn filename() -> ~str {
        ~"proto://" + *self.name
    }

    fn num_states() -> uint { self.states.len() }

    fn has_ty_params() -> bool {
        for self.states.each |s| {
            if s.ty_params.len() > 0 {
                return true;
            }
        }
        false
    }
    fn is_bounded() -> bool {
        let bounded = self.bounded.get();
        bounded
        //if bounded && self.has_ty_params() {
        //    debug!{"protocol %s has is bounded, but type parameters\
        //            are not yet supported.",
        //           *self.name};
        //    false
        //}
        //else { bounded }
    }
}

impl protocol {
    fn add_state(name: ident, dir: direction) -> state {
        self.add_state_poly(name, dir, ~[])
    }

    fn add_state_poly(name: ident, dir: direction,
                      +ty_params: ~[ast::ty_param]) -> state {
        let messages = dvec();

        let state = state_(@{
            id: self.states.len(),
            name: name,
            span: self.span,
            dir: dir,
            ty_params: ty_params,
            messages: messages,
            proto: self
        });

        self.states.push(state);
        state
    }
}

trait visitor<Tproto, Tstate, Tmessage> {
    fn visit_proto(proto: protocol, st: &[Tstate]) -> Tproto;
    fn visit_state(state: state, m: &[Tmessage]) -> Tstate;
    fn visit_message(name: ident, spane: span, tys: &[@ast::ty],
                     this: state, next: next_state) -> Tmessage;
}

fn visit<Tproto, Tstate, Tmessage, V: visitor<Tproto, Tstate, Tmessage>>(
    proto: protocol, visitor: V) -> Tproto {

    // the copy keywords prevent recursive use of dvec
    let states = do (copy proto.states).map_to_vec |s| {
        let messages = do (copy s.messages).map_to_vec |m| {
            let message(name, span, tys, this, next) = m;
            visitor.visit_message(name, span, tys, this, next)
        };
        visitor.visit_state(s, messages)
    };
    visitor.visit_proto(proto, states)
}
