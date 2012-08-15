// A protocol compiler for Rust.

import to_str::ToStr;

import dvec::dvec;

import ast::ident;
import util::interner;
import print::pprust;
import pprust::{item_to_str, ty_to_str};
import ext::base::{mk_ctxt, ext_ctxt};
import parse;
import parse::*;
import proto::*;

import ast_builder::append_types;
import ast_builder::path;

// Transitional reexports so qquote can find the paths it is looking for
mod syntax {
    import ext;
    export ext;
    import parse;
    export parse;
}

trait gen_send {
    fn gen_send(cx: ext_ctxt, try: bool) -> @ast::item;
}

trait to_type_decls {
    fn to_type_decls(cx: ext_ctxt) -> ~[@ast::item];
    fn to_endpoint_decls(cx: ext_ctxt, dir: direction) -> ~[@ast::item];
}

trait gen_init {
    fn gen_init(cx: ext_ctxt) -> @ast::item;
    fn compile(cx: ext_ctxt) -> @ast::item;
}

impl message: gen_send {
    fn gen_send(cx: ext_ctxt, try: bool) -> @ast::item {
        debug!{"pipec: gen_send"};
        match self {
          message(id, span, tys, this,
                  some({state: next, tys: next_tys})) => {
            debug!("pipec: next state exists");
            let next = this.proto.get_state(next);
            assert next_tys.len() == next.ty_params.len();
            let arg_names = tys.mapi(|i, _ty| @(~"x_" + i.to_str()));

            let args_ast = (arg_names, tys).map(
                |n, t| cx.arg_mode(n, t, ast::by_copy)
            );

            let pipe_ty = cx.ty_path_ast_builder(
                path(this.data_name(), span)
                .add_tys(cx.ty_vars(this.ty_params)));
            let args_ast = vec::append(
                ~[cx.arg_mode(@~"pipe",
                              pipe_ty,
                              ast::by_copy)],
                args_ast);

            let mut body = ~"{\n";

            if this.proto.is_bounded() {
                let (sp, rp) = match (this.dir, next.dir) {
                  (send, send) => (~"c", ~"s"),
                  (send, recv) => (~"s", ~"c"),
                  (recv, send) => (~"s", ~"c"),
                  (recv, recv) => (~"c", ~"s")
                };

                body += ~"let b = pipe.reuse_buffer();\n";
                body += fmt!("let %s = pipes::send_packet_buffered(\
                              ptr::addr_of(b.buffer.data.%s));\n",
                             sp, *next.name);
                body += fmt!("let %s = pipes::recv_packet_buffered(\
                              ptr::addr_of(b.buffer.data.%s));\n",
                             rp, *next.name);
            }
            else {
                let pat = match (this.dir, next.dir) {
                  (send, send) => "(c, s)",
                  (send, recv) => "(s, c)",
                  (recv, send) => "(s, c)",
                  (recv, recv) => "(c, s)"
                };

                body += fmt!("let %s = pipes::entangle();\n", pat);
            }
            body += fmt!("let message = %s::%s(%s);\n",
                         *this.proto.name,
                         *self.name(),
                         str::connect(vec::append_one(arg_names, @~"s")
                                      .map(|x| *x),
                                      ~", "));

            if !try {
                body += fmt!{"pipes::send(pipe, message);\n"};
                // return the new channel
                body += ~"c }";
            }
            else {
                body += fmt!("if pipes::send(pipe, message) {\n \
                                  some(c) \
                              } else { none } }");
            }

            let body = cx.parse_expr(body);

            let mut rty = cx.ty_path_ast_builder(path(next.data_name(),
                                                      span)
                                                 .add_tys(next_tys));
            if try {
                rty = cx.ty_option(rty);
            }

            let name = if try {
                @(~"try_" + *self.name())
            }
            else { self.name() };

            cx.item_fn_poly(name,
                            args_ast,
                            rty,
                            self.get_params(),
                            cx.expr_block(body))
          }

            message(id, span, tys, this, none) => {
                debug!{"pipec: no next state"};
                let arg_names = tys.mapi(|i, _ty| @(~"x_" + i.to_str()));

                let args_ast = (arg_names, tys).map(
                    |n, t| cx.arg_mode(n, t, ast::by_copy)
                );

                let args_ast = vec::append(
                    ~[cx.arg_mode(@~"pipe",
                                  cx.ty_path_ast_builder(
                                      path(this.data_name(), span)
                                      .add_tys(cx.ty_vars(this.ty_params))),
                                  ast::by_copy)],
                    args_ast);

                let message_args = if arg_names.len() == 0 {
                    ~""
                }
                else {
                    ~"(" + str::connect(arg_names.map(|x| *x), ~", ") + ~")"
                };

                let mut body = ~"{ ";
                body += fmt!{"let message = %s::%s%s;\n",
                             *this.proto.name,
                             *self.name(),
                             message_args};

                if !try {
                    body += fmt!{"pipes::send(pipe, message);\n"};
                    body += ~" }";
                } else {
                    body += fmt!("if pipes::send(pipe, message) { \
                                      some(()) \
                                  } else { none } }");
                }

                let body = cx.parse_expr(body);

                let name = if try {
                    @(~"try_" + *self.name())
                }
                else { self.name() };

                cx.item_fn_poly(name,
                                args_ast,
                                if try {
                                    cx.ty_option(cx.ty_nil_ast_builder())
                                } else {
                                    cx.ty_nil_ast_builder()
                                },
                                self.get_params(),
                                cx.expr_block(body))
            }
          }
        }

    fn to_ty(cx: ext_ctxt) -> @ast::ty {
        cx.ty_path_ast_builder(path(self.name(), self.span())
          .add_tys(cx.ty_vars(self.get_params())))
    }
}

impl state: to_type_decls {
    fn to_type_decls(cx: ext_ctxt) -> ~[@ast::item] {
        debug!{"pipec: to_type_decls"};
        // This compiles into two different type declarations. Say the
        // state is called ping. This will generate both `ping` and
        // `ping_message`. The first contains data that the user cares
        // about. The second is the same thing, but extended with a
        // next packet pointer, which is used under the covers.

        let name = self.data_name();

        let mut items_msg = ~[];

        for self.messages.each |m| {
            let message(name, _span, tys, this, next) = m;

            let tys = match next {
              some({state: next, tys: next_tys}) => {
                let next = this.proto.get_state(next);
                let next_name = next.data_name();

                let dir = match this.dir {
                  send => @~"server",
                  recv => @~"client"
                };

                vec::append_one(tys,
                                cx.ty_path_ast_builder((dir + next_name)
                                           .add_tys(next_tys)))
              }
              none => tys
            };

            let v = cx.variant(name, tys);

            vec::push(items_msg, v);
        }

        ~[cx.item_enum_poly(name,
                            ast::enum_def({ variants: items_msg,
                                            common: none }),
                            self.ty_params)]
    }

    fn to_endpoint_decls(cx: ext_ctxt, dir: direction) -> ~[@ast::item] {
        debug!{"pipec: to_endpoint_decls"};
        let dir = match dir {
          send => (*self).dir,
          recv => (*self).dir.reverse()
        };
        let mut items = ~[];
        for self.messages.each |m| {
            if dir == send {
                vec::push(items, m.gen_send(cx, true));
                vec::push(items, m.gen_send(cx, false));
            }
        }

        if !self.proto.is_bounded() {
            vec::push(items,
                      cx.item_ty_poly(
                          self.data_name(),
                          cx.ty_path_ast_builder(
                              (@~"pipes" + @(dir.to_str() + ~"_packet"))
                              .add_ty(cx.ty_path_ast_builder(
                                  (self.proto.name + self.data_name())
                                  .add_tys(cx.ty_vars(self.ty_params))))),
                          self.ty_params));
        }
        else {
            vec::push(items,
                      cx.item_ty_poly(
                          self.data_name(),
                          cx.ty_path_ast_builder(
                              (@~"pipes" + @(dir.to_str()
                                             + ~"_packet_buffered"))
                              .add_tys(~[cx.ty_path_ast_builder(
                                  (self.proto.name + self.data_name())
                                  .add_tys(cx.ty_vars(self.ty_params))),
                                         self.proto.buffer_ty_path(cx)])),
                          self.ty_params));
        };
        items
    }
}

impl protocol: gen_init {
    fn gen_init(cx: ext_ctxt) -> @ast::item {
        let ext_cx = cx;

        debug!{"gen_init"};
        let start_state = self.states[0];

        let body = if !self.is_bounded() {
            match start_state.dir {
              send => #ast { pipes::entangle() },
              recv => {
                #ast {{
                    let (s, c) = pipes::entangle();
                    (c, s)
                }}
              }
            }
        }
        else {
            let body = self.gen_init_bounded(ext_cx);
            match start_state.dir {
              send => body,
              recv => {
                #ast {{
                    let (s, c) = $(body);
                    (c, s)
                }}
              }
            }
        };

        cx.parse_item(fmt!{"fn init%s() -> (client::%s, server::%s)\
                            { import pipes::has_buffer; %s }",
                           start_state.ty_params.to_source(),
                           start_state.to_ty(cx).to_source(),
                           start_state.to_ty(cx).to_source(),
                           body.to_source()})
    }

    fn gen_buffer_init(ext_cx: ext_ctxt) -> @ast::expr {
        ext_cx.rec(self.states.map_to_vec(|s| {
            let fty = s.to_ty(ext_cx);
            ext_cx.field_imm(s.name, #ast { pipes::mk_packet::<$(fty)>() })
        }))
    }

    fn gen_init_bounded(ext_cx: ext_ctxt) -> @ast::expr {
        debug!{"gen_init_bounded"};
        let buffer_fields = self.gen_buffer_init(ext_cx);

        let buffer = #ast {
            ~{header: pipes::buffer_header(),
              data: $(buffer_fields)}
        };

        let entangle_body = ext_cx.block_expr(
            ext_cx.block(
                self.states.map_to_vec(
                    |s| ext_cx.parse_stmt(
                        fmt!{"data.%s.set_buffer(buffer)", *s.name})),
                ext_cx.parse_expr(
                    fmt!{"ptr::addr_of(data.%s)", *self.states[0].name})));

        #ast {{
            let buffer = $(buffer);
            do pipes::entangle_buffer(buffer) |buffer, data| {
                $(entangle_body)
            }
        }}
    }

    fn buffer_ty_path(cx: ext_ctxt) -> @ast::ty {
        let mut params: ~[ast::ty_param] = ~[];
        for (copy self.states).each |s| {
            for s.ty_params.each |tp| {
                match params.find(|tpp| *tp.ident == *tpp.ident) {
                  none => vec::push(params, tp),
                  _ => ()
                }
            }
        }

        cx.ty_path_ast_builder(path(@~"__Buffer", self.span)
                               .add_tys(cx.ty_vars(params)))
    }

    fn gen_buffer_type(cx: ext_ctxt) -> @ast::item {
        let ext_cx = cx;
        let mut params: ~[ast::ty_param] = ~[];
        let fields = do (copy self.states).map_to_vec |s| {
            for s.ty_params.each |tp| {
                match params.find(|tpp| *tp.ident == *tpp.ident) {
                  none => vec::push(params, tp),
                  _ => ()
                }
            }
            let ty = s.to_ty(cx);
            let fty = #ast[ty] {
                pipes::packet<$(ty)>
            };
            cx.ty_field_imm(s.name, fty)
        };

        cx.item_ty_poly(
            @~"__Buffer",
            cx.ty_rec(fields),
            params)
    }

    fn compile(cx: ext_ctxt) -> @ast::item {
        let mut items = ~[self.gen_init(cx)];
        let mut client_states = ~[];
        let mut server_states = ~[];

        // :(
        for (copy self.states).each |s| {
            items += s.to_type_decls(cx);

            client_states += s.to_endpoint_decls(cx, send);
            server_states += s.to_endpoint_decls(cx, recv);
        }

        if self.is_bounded() {
            vec::push(items, self.gen_buffer_type(cx))
        }

        vec::push(items,
                  cx.item_mod(@~"client",
                              client_states));
        vec::push(items,
                  cx.item_mod(@~"server",
                              server_states));

        cx.item_mod(self.name, items)
    }
}

trait to_source {
    // Takes a thing and generates a string containing rust code for it.
    fn to_source() -> ~str;
}

impl @ast::item: to_source {
    fn to_source() -> ~str {
        item_to_str(self)
    }
}

impl ~[@ast::item]: to_source {
    fn to_source() -> ~str {
        str::connect(self.map(|i| i.to_source()), ~"\n\n")
    }
}

impl @ast::ty: to_source {
    fn to_source() -> ~str {
        ty_to_str(self)
    }
}

impl ~[@ast::ty]: to_source {
    fn to_source() -> ~str {
        str::connect(self.map(|i| i.to_source()), ~", ")
    }
}

impl ~[ast::ty_param]: to_source {
    fn to_source() -> ~str {
        pprust::typarams_to_str(self)
    }
}

impl @ast::expr: to_source {
    fn to_source() -> ~str {
        pprust::expr_to_str(self)
    }
}

trait ext_ctxt_parse_utils {
    fn parse_item(s: ~str) -> @ast::item;
    fn parse_expr(s: ~str) -> @ast::expr;
    fn parse_stmt(s: ~str) -> @ast::stmt;
}

impl ext_ctxt: ext_ctxt_parse_utils {
    fn parse_item(s: ~str) -> @ast::item {
        let res = parse::parse_item_from_source_str(
            ~"***protocol expansion***",
            @(copy s),
            self.cfg(),
            ~[],
            self.parse_sess());
        match res {
          some(ast) => ast,
          none => {
            error!{"Parse error with ```\n%s\n```", s};
            fail
          }
        }
    }

    fn parse_stmt(s: ~str) -> @ast::stmt {
        parse::parse_stmt_from_source_str(
            ~"***protocol expansion***",
            @(copy s),
            self.cfg(),
            ~[],
            self.parse_sess())
    }

    fn parse_expr(s: ~str) -> @ast::expr {
        parse::parse_expr_from_source_str(
            ~"***protocol expansion***",
            @(copy s),
            self.cfg(),
            self.parse_sess())
    }
}
