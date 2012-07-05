// A protocol compiler for Rust.

import to_str::to_str;

import dvec::dvec;
import dvec::extensions;

import ast::ident;
import util::interner;
import interner::{intern, get};
import print::pprust;
import pprust::{item_to_str, ty_to_str};
import ext::base::{mk_ctxt, ext_ctxt};
import parse;
import parse::{parse_item_from_source_str};

import ast_builder::ast_builder;
import ast_builder::methods;
import ast_builder::path;

enum direction {
    send, recv
}

impl of to_str for direction {
    fn to_str() -> str {
        alt self {
          send { "send" }
          recv { "recv" }
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

enum message {
    // name, data, current state, next state, next tys
    message(ident, ~[@ast::ty], state, state, ~[@ast::ty])
}

impl methods for message {
    fn name() -> ident {
        alt self {
          message(id, _, _, _, _) {
            id
          }
        }
    }

    // Return the type parameters actually used by this message
    fn get_params() -> ~[ast::ty_param] {
        let mut used = ~[];
        alt self {
          message(_, tys, this, _, next_tys) {
            let parms = this.ty_params;
            for vec::append(tys, next_tys).each |ty| {
                alt ty.node {
                  ast::ty_path(path, _) {
                    if path.idents.len() == 1 {
                        let id = path.idents[0];

                        let found = parms.find(|p| id == p.ident);

                        alt found {
                          some(p) {
                            if !used.contains(p) {
                                vec::push(used, p);
                            }
                          }
                          none { }
                        }
                    }
                  }
                  _ { }
                }
            }
          }
        }
        used
    }

    fn gen_send(cx: ext_ctxt) -> @ast::item {
        alt self {
          message(id, tys, this, next, next_tys) {
            let arg_names = tys.mapi(|i, _ty| @("x_" + i.to_str()));

            let args = (arg_names, tys).map(|n, t|
                                            *n + ": " + t.to_source());

            let args_ast = (arg_names, tys).map(
                |n, t| cx.arg_mode(n, t, ast::by_copy)
            );

            let args_ast = vec::append(
                ~[cx.arg_mode(@"pipe",
                              cx.ty_path(path(this.data_name())
                                        .add_tys(cx.ty_vars(this.ty_params))),
                              ast::by_copy)],
                args_ast);

            let pat = alt (this.dir, next.dir) {
              (send, send) { "(c, s)" }
              (send, recv) { "(s, c)" }
              (recv, send) { "(s, c)" }
              (recv, recv) { "(c, s)" }
            };

            let mut body = #fmt("{ let %s = pipes::entangle();\n", pat);
            body += #fmt("let message = %s::%s(%s);\n",
                         *this.proto.name,
                         *self.name(),
                         str::connect(vec::append_one(arg_names, @"s")
                                      .map(|x| *x),
                                      ", "));
            body += #fmt("pipes::send(pipe, message);\n");
            body += "c }";

            let body = cx.parse_expr(body);

            cx.item_fn_poly(self.name(),
                            args_ast,
                            cx.ty_path(path(next.data_name())
                                      .add_tys(next_tys)),
                            self.get_params(),
                            cx.expr_block(body))
          }
        }
    }
}

enum state {
    state_(@{
        name: ident,
        dir: direction,
        ty_params: ~[ast::ty_param],
        messages: dvec<message>,
        proto: protocol,
    }),
}

impl methods for state {
    fn add_message(name: ident, +data: ~[@ast::ty], next: state,
                   +next_tys: ~[@ast::ty]) {
        assert next_tys.len() == next.ty_params.len();
        self.messages.push(message(name, data, self, next, next_tys));
    }

    fn filename() -> str {
        (*self).proto.filename()
    }

    fn data_name() -> ident {
        self.name
    }

    fn to_ty(cx: ext_ctxt) -> @ast::ty {
        cx.ty_path(path(self.name).add_tys(cx.ty_vars(self.ty_params)))
    }

    fn to_type_decls(cx: ext_ctxt) -> [@ast::item]/~ {
        // This compiles into two different type declarations. Say the
        // state is called ping. This will generate both `ping` and
        // `ping_message`. The first contains data that the user cares
        // about. The second is the same thing, but extended with a
        // next packet pointer, which is used under the covers.

        let name = self.data_name();

        let mut items_msg = []/~;

        for self.messages.each |m| {
            let message(_, tys, this, next, next_tys) = m;

            let name = m.name();
            let next_name = next.data_name();

            let dir = alt this.dir {
              send { @"server" }
              recv { @"client" }
            };

            let v = cx.variant(name,
                               vec::append_one(
                                   tys,
                                   cx.ty_path((dir + next_name)
                                              .add_tys(next_tys))));

            vec::push(items_msg, v);
        }

        ~[cx.item_enum_poly(name, items_msg, self.ty_params)]
    }

    fn to_endpoint_decls(cx: ext_ctxt, dir: direction) -> [@ast::item]/~ {
        let dir = alt dir {
          send { (*self).dir }
          recv { (*self).dir.reverse() }
        };
        let mut items = ~[];
        for self.messages.each |m| {
            if dir == send {
                vec::push(items, m.gen_send(cx))
            }
        }

        vec::push(items,
                  cx.item_ty_poly(
                      self.data_name(),
                      cx.ty_path(
                          (@"pipes" + @(dir.to_str() + "_packet"))
                          .add_ty(cx.ty_path(
                              (self.proto.name + self.data_name())
                              .add_tys(cx.ty_vars(self.ty_params))))),
                      self.ty_params));
        items
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

    fn add_state_poly(name: ident, dir: direction,
                      +ty_params: ~[ast::ty_param]) -> state {
        let messages = dvec();

        let state = state_(@{
            name: name,
            dir: dir,
            ty_params: ty_params,
            messages: messages,
            proto: self
        });

        self.states.push(state);
        state
    }

    fn filename() -> str {
        "proto://" + *self.name
    }

    fn gen_init(cx: ext_ctxt) -> @ast::item {
        let start_state = self.states[0];

        let body = alt start_state.dir {
          send { cx.parse_expr("pipes::entangle()") }
          recv {
            cx.parse_expr("{ \
                           let (s, c) = pipes::entangle(); \
                           (c, s) \
                           }")
          }
        };

        parse_item_from_source_str(
            self.filename(),
            @#fmt("fn init%s() -> (client::%s, server::%s)\
                   { %s }",
                  start_state.ty_params.to_source(),
                  start_state.to_ty(cx).to_source(),
                  start_state.to_ty(cx).to_source(),
                  body.to_source()),
            cx.cfg(),
            []/~,
            ast::public,
            cx.parse_sess()).get()
    }

    fn compile(cx: ext_ctxt) -> @ast::item {
        let mut items = ~[self.gen_init(cx)];
        let mut client_states = ~[];
        let mut server_states = ~[];

        for self.states.each |s| {
            items += s.to_type_decls(cx);

            client_states += s.to_endpoint_decls(cx, send);
            server_states += s.to_endpoint_decls(cx, recv);
        }

        vec::push(items,
                  cx.item_mod(@"client",
                              client_states));
        vec::push(items,
                  cx.item_mod(@"server",
                              server_states));

        cx.item_mod(self.name, items)
    }
}

iface to_source {
    // Takes a thing and generates a string containing rust code for it.
    fn to_source() -> str;
}

impl of to_source for @ast::item {
    fn to_source() -> str {
        item_to_str(self)
    }
}

impl of to_source for [@ast::item]/~ {
    fn to_source() -> str {
        str::connect(self.map(|i| i.to_source()), "\n\n")
    }
}

impl of to_source for @ast::ty {
    fn to_source() -> str {
        ty_to_str(self)
    }
}

impl of to_source for [@ast::ty]/~ {
    fn to_source() -> str {
        str::connect(self.map(|i| i.to_source()), ", ")
    }
}

impl of to_source for ~[ast::ty_param] {
    fn to_source() -> str {
        pprust::typarams_to_str(self)
    }
}

impl of to_source for @ast::expr {
    fn to_source() -> str {
        pprust::expr_to_str(self)
    }
}

impl parse_utils for ext_ctxt {
    fn parse_item(s: str) -> @ast::item {
        let res = parse::parse_item_from_source_str(
            "***protocol expansion***",
            @(copy s),
            self.cfg(),
            []/~,
            ast::public,
            self.parse_sess());
        alt res {
          some(ast) { ast }
          none {
            #error("Parse error with ```\n%s\n```", s);
            fail
          }
        }
    }

    fn parse_expr(s: str) -> @ast::expr {
        parse::parse_expr_from_source_str(
            "***protocol expansion***",
            @(copy s),
            self.cfg(),
            self.parse_sess())
    }
}

impl methods<A: copy, B: copy> for ([A]/~, [B]/~) {
    fn zip() -> [(A, B)]/~ {
        let (a, b) = self;
        vec::zip(a, b)
    }

    fn map<C>(f: fn(A, B) -> C) -> [C]/~ {
        let (a, b) = self;
        vec::map2(a, b, f)
    }
}
