// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// A protocol compiler for Rust.

use ast;
use codemap::{dummy_sp, spanned};
use ext::base::ext_ctxt;
use ext::pipes::ast_builder::{append_types, ext_ctxt_ast_builder, path};
use ext::pipes::ast_builder::{path_global};
use ext::pipes::proto::*;
use ext::quote::rt::*;
use opt_vec;
use opt_vec::OptVec;

use core::prelude::*;
use core::str;
use core::to_str::ToStr;
use core::vec;

pub trait gen_send {
    fn gen_send(&mut self, cx: @ext_ctxt, try: bool) -> @ast::item;
    fn to_ty(&mut self, cx: @ext_ctxt) -> @ast::Ty;
}

pub trait to_type_decls {
    fn to_type_decls(&self, cx: @ext_ctxt) -> ~[@ast::item];
    fn to_endpoint_decls(&self, cx: @ext_ctxt,
                         dir: direction) -> ~[@ast::item];
}

pub trait gen_init {
    fn gen_init(&self, cx: @ext_ctxt) -> @ast::item;
    fn compile(&self, cx: @ext_ctxt) -> @ast::item;
    fn buffer_ty_path(&self, cx: @ext_ctxt) -> @ast::Ty;
    fn gen_buffer_type(&self, cx: @ext_ctxt) -> @ast::item;
    fn gen_buffer_init(&self, ext_cx: @ext_ctxt) -> @ast::expr;
    fn gen_init_bounded(&self, ext_cx: @ext_ctxt) -> @ast::expr;
}

impl gen_send for message {
    fn gen_send(&mut self, cx: @ext_ctxt, try: bool) -> @ast::item {
        debug!("pipec: gen_send");
        let name = self.name();

        match *self {
          message(ref _id, span, ref tys, this, Some(ref next_state)) => {
            debug!("pipec: next state exists");
            let next = this.proto.get_state(next_state.state);
            assert!(next_state.tys.len() ==
                next.generics.ty_params.len());
            let arg_names = tys.mapi(|i, _ty| cx.ident_of(~"x_"+i.to_str()));
            let args_ast = vec::map_zip(arg_names, *tys, |n, t| cx.arg(*n, *t));

            let pipe_ty = cx.ty_path_ast_builder(
                path(~[this.data_name()], span)
                .add_tys(cx.ty_vars_global(&this.generics.ty_params)));
            let args_ast = vec::append(
                ~[cx.arg(cx.ident_of(~"pipe"),
                              pipe_ty)],
                args_ast);

            let mut body = ~"{\n";
            body += fmt!("use super::%s;\n", name);

            if this.proto.is_bounded() {
                let (sp, rp) = match (this.dir, next.dir) {
                  (send, send) => (~"c", ~"s"),
                  (send, recv) => (~"s", ~"c"),
                  (recv, send) => (~"s", ~"c"),
                  (recv, recv) => (~"c", ~"s")
                };

                body += ~"let b = pipe.reuse_buffer();\n";
                body += fmt!("let %s = ::core::pipes::SendPacketBuffered(\
                              ::ptr::addr_of(&(b.buffer.data.%s)));\n",
                             sp, next.name);
                body += fmt!("let %s = ::core::pipes::RecvPacketBuffered(\
                              ::ptr::addr_of(&(b.buffer.data.%s)));\n",
                             rp, next.name);
            }
            else {
                let pat = match (this.dir, next.dir) {
                  (send, send) => "(c, s)",
                  (send, recv) => "(s, c)",
                  (recv, send) => "(s, c)",
                  (recv, recv) => "(c, s)"
                };

                body += fmt!("let %s = ::core::pipes::entangle();\n", pat);
            }
            body += fmt!("let message = %s(%s);\n",
                         name,
                         str::connect(vec::append_one(
                           arg_names.map(|x| cx.str_of(*x)),
                             ~"s"), ~", "));

            if !try {
                body += fmt!("::core::pipes::send(pipe, message);\n");
                // return the new channel
                body += ~"c }";
            }
            else {
                body += fmt!("if ::core::pipes::send(pipe, message) {\n \
                                  ::core::pipes::rt::make_some(c) \
                              } else { ::core::pipes::rt::make_none() } }");
            }

            let body = cx.parse_expr(body);

            let mut rty = cx.ty_path_ast_builder(path(~[next.data_name()],
                                                      span)
                                               .add_tys(copy next_state.tys));
            if try {
                rty = cx.ty_option(rty);
            }

            let name = cx.ident_of(if try { ~"try_" + name } else { name } );

            cx.item_fn_poly(name,
                            args_ast,
                            rty,
                            self.get_generics(),
                            cx.expr_block(body))
          }

            message(ref _id, span, ref tys, this, None) => {
                debug!("pipec: no next state");
                let arg_names = tys.mapi(|i, _ty| (~"x_" + i.to_str()));

                let args_ast = do vec::map_zip(arg_names, *tys) |n, t| {
                    cx.arg(cx.ident_of(*n), *t)
                };

                let args_ast = vec::append(
                    ~[cx.arg(cx.ident_of(~"pipe"),
                             cx.ty_path_ast_builder(
                                 path(~[this.data_name()], span)
                                 .add_tys(cx.ty_vars_global(
                                     &this.generics.ty_params))))],
                    args_ast);

                let message_args = if arg_names.len() == 0 {
                    ~""
                }
                else {
                    ~"(" + str::connect(arg_names.map(|x| copy *x),
                                        ~", ") + ~")"
                };

                let mut body = ~"{ ";
                body += fmt!("use super::%s;\n", name);
                body += fmt!("let message = %s%s;\n", name, message_args);

                if !try {
                    body += fmt!("::core::pipes::send(pipe, message);\n");
                    body += ~" }";
                } else {
                    body += fmt!("if ::core::pipes::send(pipe, message) \
                                        { \
                                      ::core::pipes::rt::make_some(()) \
                                  } else { \
                                    ::core::pipes::rt::make_none() \
                                  } }");
                }

                let body = cx.parse_expr(body);

                let name = if try { ~"try_" + name } else { name };

                cx.item_fn_poly(cx.ident_of(name),
                                args_ast,
                                if try {
                                    cx.ty_option(cx.ty_nil_ast_builder())
                                } else {
                                    cx.ty_nil_ast_builder()
                                },
                                self.get_generics(),
                                cx.expr_block(body))
            }
          }
        }

    fn to_ty(&mut self, cx: @ext_ctxt) -> @ast::Ty {
        cx.ty_path_ast_builder(path(~[cx.ident_of(self.name())], self.span())
          .add_tys(cx.ty_vars_global(&self.get_generics().ty_params)))
    }
}

impl to_type_decls for state {
    fn to_type_decls(&self, cx: @ext_ctxt) -> ~[@ast::item] {
        debug!("pipec: to_type_decls");
        // This compiles into two different type declarations. Say the
        // state is called ping. This will generate both `ping` and
        // `ping_message`. The first contains data that the user cares
        // about. The second is the same thing, but extended with a
        // next packet pointer, which is used under the covers.

        let name = self.data_name();

        let mut items_msg = ~[];

        for self.messages.each |m| {
            let message(name, span, tys, this, next) = copy *m;

            let tys = match next {
              Some(ref next_state) => {
                let next = this.proto.get_state((next_state.state));
                let next_name = cx.str_of(next.data_name());

                let dir = match this.dir {
                  send => ~"server",
                  recv => ~"client"
                };

                vec::append_one(tys,
                                cx.ty_path_ast_builder(
                                    path(~[cx.ident_of(dir),
                                           cx.ident_of(next_name)], span)
                                    .add_tys(copy next_state.tys)))
              }
              None => tys
            };

            let v = cx.variant(cx.ident_of(name), span, tys);

            items_msg.push(v);
        }

        ~[
            cx.item_enum_poly(
                name,
                self.span,
                ast::enum_def { variants: items_msg },
                cx.strip_bounds(&self.generics)
            )
        ]
    }

    fn to_endpoint_decls(&self, cx: @ext_ctxt,
                         dir: direction) -> ~[@ast::item] {
        debug!("pipec: to_endpoint_decls");
        let dir = match dir {
          send => (*self).dir,
          recv => (*self).dir.reverse()
        };
        let mut items = ~[];

        {
            let messages = &mut *self.messages;
            for vec::each_mut(*messages) |m| {
                if dir == send {
                    items.push(m.gen_send(cx, true));
                    items.push(m.gen_send(cx, false));
                }
            }
        }

        if !self.proto.is_bounded() {
            items.push(
                cx.item_ty_poly(
                    self.data_name(),
                    self.span,
                    cx.ty_path_ast_builder(
                        path_global(~[cx.ident_of(~"core"),
                                      cx.ident_of(~"pipes"),
                                      cx.ident_of(dir.to_str() + ~"Packet")],
                             dummy_sp())
                        .add_ty(cx.ty_path_ast_builder(
                            path(~[cx.ident_of(~"super"),
                                   self.data_name()],
                                 dummy_sp())
                            .add_tys(cx.ty_vars_global(
                                &self.generics.ty_params))))),
                    cx.strip_bounds(&self.generics)));
        }
        else {
            items.push(
                cx.item_ty_poly(
                    self.data_name(),
                    self.span,
                    cx.ty_path_ast_builder(
                        path_global(~[cx.ident_of(~"core"),
                                      cx.ident_of(~"pipes"),
                                      cx.ident_of(dir.to_str()
                                                  + ~"PacketBuffered")],
                             dummy_sp())
                        .add_tys(~[cx.ty_path_ast_builder(
                            path(~[cx.ident_of(~"super"),
                                   self.data_name()],
                                        dummy_sp())
                            .add_tys(cx.ty_vars_global(
                                &self.generics.ty_params))),
                                   self.proto.buffer_ty_path(cx)])),
                    cx.strip_bounds(&self.generics)));
        };
        items
    }
}

impl gen_init for protocol {
    fn gen_init(&self, cx: @ext_ctxt) -> @ast::item {
        let ext_cx = cx;

        debug!("gen_init");
        let start_state = self.states[0];

        let body = if !self.is_bounded() {
            match start_state.dir {
              send => quote_expr!( ::core::pipes::entangle() ),
              recv => {
                quote_expr!({
                    let (s, c) = ::core::pipes::entangle();
                    (c, s)
                })
              }
            }
        }
        else {
            let body = self.gen_init_bounded(ext_cx);
            match start_state.dir {
              send => body,
              recv => {
                  quote_expr!({
                      let (s, c) = $body;
                      (c, s)
                  })
              }
            }
        };

        cx.parse_item(fmt!("pub fn init%s() -> (client::%s, server::%s)\
                            { pub use core::pipes::HasBuffer; %s }",
                           start_state.generics.to_source(cx),
                           start_state.to_ty(cx).to_source(cx),
                           start_state.to_ty(cx).to_source(cx),
                           body.to_source(cx)))
    }

    fn gen_buffer_init(&self, ext_cx: @ext_ctxt) -> @ast::expr {
        ext_cx.struct_expr(path(~[ext_cx.ident_of(~"__Buffer")],
                                dummy_sp()),
                      self.states.map_to_vec(|s| {
            let fty = s.to_ty(ext_cx);
            ext_cx.field_imm(ext_cx.ident_of(s.name),
                             quote_expr!(
                                 ::core::pipes::mk_packet::<$fty>()
                             ))
        }))
    }

    fn gen_init_bounded(&self, ext_cx: @ext_ctxt) -> @ast::expr {
        debug!("gen_init_bounded");
        let buffer_fields = self.gen_buffer_init(ext_cx);
        let buffer = quote_expr!(~::core::pipes::Buffer {
            header: ::core::pipes::BufferHeader(),
            data: $buffer_fields,
        });

        let entangle_body = ext_cx.block_expr(
            ext_cx.block(
                self.states.map_to_vec(
                    |s| ext_cx.parse_stmt(
                        fmt!("data.%s.set_buffer(buffer)",
                             s.name))),
                ext_cx.parse_expr(
                    fmt!("::ptr::addr_of(&(data.%s))",
                         self.states[0].name))));

        quote_expr!({
            let buffer = $buffer;
            do ::core::pipes::entangle_buffer(buffer) |buffer, data| {
                $entangle_body
            }
        })
    }

    fn buffer_ty_path(&self, cx: @ext_ctxt) -> @ast::Ty {
        let mut params: OptVec<ast::TyParam> = opt_vec::Empty;
        for (copy self.states).each |s| {
            for s.generics.ty_params.each |tp| {
                match params.find(|tpp| tp.ident == tpp.ident) {
                  None => params.push(*tp),
                  _ => ()
                }
            }
        }

        cx.ty_path_ast_builder(path(~[cx.ident_of(~"super"),
                                      cx.ident_of(~"__Buffer")],
                                    copy self.span)
                               .add_tys(cx.ty_vars_global(&params)))
    }

    fn gen_buffer_type(&self, cx: @ext_ctxt) -> @ast::item {
        let ext_cx = cx;
        let mut params: OptVec<ast::TyParam> = opt_vec::Empty;
        let fields = do (copy self.states).map_to_vec |s| {
            for s.generics.ty_params.each |tp| {
                match params.find(|tpp| tp.ident == tpp.ident) {
                  None => params.push(*tp),
                  _ => ()
                }
            }

            let ty = s.to_ty(cx);
            let fty = quote_ty!( ::core::pipes::Packet<$ty> );

            @spanned {
                node: ast::struct_field_ {
                    kind: ast::named_field(
                            cx.ident_of(s.name),
                            ast::struct_immutable,
                            ast::inherited),
                    id: cx.next_id(),
                    ty: fty
                },
                span: dummy_sp()
            }
        };

        let generics = Generics {
            lifetimes: opt_vec::Empty,
            ty_params: params
        };

        cx.item_struct_poly(
            cx.ident_of(~"__Buffer"),
            dummy_sp(),
            ast::struct_def {
                fields: fields,
                dtor: None,
                ctor_id: None
            },
            cx.strip_bounds(&generics))
    }

    fn compile(&self, cx: @ext_ctxt) -> @ast::item {
        let mut items = ~[self.gen_init(cx)];
        let mut client_states = ~[];
        let mut server_states = ~[];

        for (copy self.states).each |s| {
            items += s.to_type_decls(cx);

            client_states += s.to_endpoint_decls(cx, send);
            server_states += s.to_endpoint_decls(cx, recv);
        }

        if self.is_bounded() {
            items.push(self.gen_buffer_type(cx))
        }

        items.push(cx.item_mod(cx.ident_of(~"client"),
                               copy self.span,
                               client_states));
        items.push(cx.item_mod(cx.ident_of(~"server"),
                               copy self.span,
                               server_states));

        cx.item_mod(cx.ident_of(self.name), copy self.span, items)
    }
}
