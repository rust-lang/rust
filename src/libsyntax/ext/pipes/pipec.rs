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

use core::prelude::*;

use ast;
use codemap::{dummy_sp, spanned};
use ext::base::ExtCtxt;
use ext::build::AstBuilder;
use ext::pipes::ast_builder::{append_types, path};
use ext::pipes::ast_builder::{path_global};
use ext::pipes::proto::*;
use ext::quote::rt::*;
use opt_vec;
use opt_vec::OptVec;

use core::vec;

pub trait gen_send {
    fn gen_send(&mut self, cx: @ExtCtxt, try: bool) -> @ast::item;
    fn to_ty(&mut self, cx: @ExtCtxt) -> @ast::Ty;
}

pub trait to_type_decls {
    fn to_type_decls(&self, cx: @ExtCtxt) -> ~[@ast::item];
    fn to_endpoint_decls(&self, cx: @ExtCtxt,
                         dir: direction) -> ~[@ast::item];
}

pub trait gen_init {
    fn gen_init(&self, cx: @ExtCtxt) -> @ast::item;
    fn compile(&self, cx: @ExtCtxt) -> @ast::item;
    fn buffer_ty_path(&self, cx: @ExtCtxt) -> @ast::Ty;
    fn gen_buffer_type(&self, cx: @ExtCtxt) -> @ast::item;
    fn gen_buffer_init(&self, ext_cx: @ExtCtxt) -> @ast::expr;
    fn gen_init_bounded(&self, ext_cx: @ExtCtxt) -> @ast::expr;
}

impl gen_send for message {
    fn gen_send(&mut self, cx: @ExtCtxt, try: bool) -> @ast::item {
        debug!("pipec: gen_send");
        let name = self.name();

        match *self {
          message(ref _id, span, ref tys, this, Some(ref next_state)) => {
            debug!("pipec: next state exists");
            let next = this.proto.get_state(next_state.state);
            assert!(next_state.tys.len() ==
                next.generics.ty_params.len());
            let arg_names = tys.mapi(|i, _ty| cx.ident_of(~"x_"+i.to_str()));
            let args_ast = vec::map_zip(arg_names, *tys, |n, t| cx.arg(span, *n, *t));

            let pipe_ty = cx.ty_path(
                path(~[this.data_name()], span)
                .add_tys(cx.ty_vars(&this.generics.ty_params)), @opt_vec::Empty);
            let args_ast = vec::append(
                ~[cx.arg(span, cx.ident_of("pipe"), pipe_ty)],
                args_ast);

            let mut body = ~"{\n";
            body += fmt!("use super::%s;\n", name);
            body += "let mut pipe = pipe;\n";

            if this.proto.is_bounded() {
                let (sp, rp) = match (this.dir, next.dir) {
                  (send, send) => (~"c", ~"s"),
                  (send, recv) => (~"s", ~"c"),
                  (recv, send) => (~"s", ~"c"),
                  (recv, recv) => (~"c", ~"s")
                };

                body += "let mut b = pipe.reuse_buffer();\n";
                body += fmt!("let %s = ::std::pipes::SendPacketBuffered(\
                              &mut (b.buffer.data.%s));\n",
                             sp, next.name);
                body += fmt!("let %s = ::std::pipes::RecvPacketBuffered(\
                              &mut (b.buffer.data.%s));\n",
                             rp, next.name);
            }
            else {
                let pat = match (this.dir, next.dir) {
                  (send, send) => "(s, c)",
                  (send, recv) => "(c, s)",
                  (recv, send) => "(c, s)",
                  (recv, recv) => "(s, c)"
                };

                body += fmt!("let %s = ::std::pipes::entangle();\n", pat);
            }
            body += fmt!("let message = %s(%s);\n",
                         name,
                         vec::append_one(
                             arg_names.map(|x| cx.str_of(*x)),
                             @"s").connect(", "));

            if !try {
                body += fmt!("::std::pipes::send(pipe, message);\n");
                // return the new channel
                body += "c }";
            }
            else {
                body += fmt!("if ::std::pipes::send(pipe, message) {\n \
                                  ::std::pipes::rt::make_some(c) \
                              } else { ::std::pipes::rt::make_none() } }");
            }

            let body = cx.parse_expr(body.to_managed());

            let mut rty = cx.ty_path(path(~[next.data_name()],
                                          span)
                                     .add_tys(copy next_state.tys), @opt_vec::Empty);
            if try {
                rty = cx.ty_option(rty);
            }

            let name = if try {cx.ident_of(~"try_" + name)} else {cx.ident_of(name)};

            cx.item_fn_poly(dummy_sp(),
                            name,
                            args_ast,
                            rty,
                            self.get_generics(),
                            cx.blk_expr(body))
          }

            message(ref _id, span, ref tys, this, None) => {
                debug!("pipec: no next state");
                let arg_names = tys.mapi(|i, _ty| (~"x_" + i.to_str()));

                let args_ast = do vec::map_zip(arg_names, *tys) |n, t| {
                    cx.arg(span, cx.ident_of(*n), *t)
                };

                let args_ast = vec::append(
                    ~[cx.arg(span,
                             cx.ident_of("pipe"),
                             cx.ty_path(
                                 path(~[this.data_name()], span)
                                 .add_tys(cx.ty_vars(
                                     &this.generics.ty_params)), @opt_vec::Empty))],
                    args_ast);

                let message_args = if arg_names.len() == 0 {
                    ~""
                }
                else {
                    ~"(" + arg_names.map(|x| copy *x).connect(", ") + ")"
                };

                let mut body = ~"{ ";
                body += fmt!("use super::%s;\n", name);
                body += fmt!("let message = %s%s;\n", name, message_args);

                if !try {
                    body += fmt!("::std::pipes::send(pipe, message);\n");
                    body += " }";
                } else {
                    body += fmt!("if ::std::pipes::send(pipe, message) \
                                        { \
                                      ::std::pipes::rt::make_some(()) \
                                  } else { \
                                    ::std::pipes::rt::make_none() \
                                  } }");
                }

                let body = cx.parse_expr(body.to_managed());

                let name = if try {cx.ident_of(~"try_" + name)} else {cx.ident_of(name)};

                cx.item_fn_poly(dummy_sp(),
                                name,
                                args_ast,
                                if try {
                                    cx.ty_option(cx.ty_nil())
                                } else {
                                    cx.ty_nil()
                                },
                                self.get_generics(),
                                cx.blk_expr(body))
            }
          }
        }

    fn to_ty(&mut self, cx: @ExtCtxt) -> @ast::Ty {
        cx.ty_path(path(~[cx.ident_of(self.name())], self.span())
          .add_tys(cx.ty_vars(&self.get_generics().ty_params)), @opt_vec::Empty)
    }
}

impl to_type_decls for state {
    fn to_type_decls(&self, cx: @ExtCtxt) -> ~[@ast::item] {
        debug!("pipec: to_type_decls");
        // This compiles into two different type declarations. Say the
        // state is called ping. This will generate both `ping` and
        // `ping_message`. The first contains data that the user cares
        // about. The second is the same thing, but extended with a
        // next packet pointer, which is used under the covers.

        let name = self.data_name();

        let mut items_msg = ~[];

        for self.messages.iter().advance |m| {
            let message(name, span, tys, this, next) = copy *m;

            let tys = match next {
              Some(ref next_state) => {
                let next = this.proto.get_state((next_state.state));
                let next_name = cx.str_of(next.data_name());

                let dir = match this.dir {
                  send => "server",
                  recv => "client"
                };

                vec::append_one(tys,
                                cx.ty_path(
                                    path(~[cx.ident_of(dir),
                                           cx.ident_of(next_name)], span)
                                    .add_tys(copy next_state.tys), @opt_vec::Empty))
              }
              None => tys
            };

            let v = cx.variant(span, cx.ident_of(name), tys);

            items_msg.push(v);
        }

        ~[
            cx.item_enum_poly(
                self.span,
                name,
                ast::enum_def { variants: items_msg },
                cx.strip_bounds(&self.generics)
            )
        ]
    }

    fn to_endpoint_decls(&self, cx: @ExtCtxt,
                         dir: direction) -> ~[@ast::item] {
        debug!("pipec: to_endpoint_decls");
        let dir = match dir {
          send => (*self).dir,
          recv => (*self).dir.reverse()
        };
        let mut items = ~[];

        {
            for self.messages.mut_iter().advance |m| {
                if dir == send {
                    items.push(m.gen_send(cx, true));
                    items.push(m.gen_send(cx, false));
                }
            }
        }

        if !self.proto.is_bounded() {
            items.push(
                cx.item_ty_poly(
                    self.span,
                    self.data_name(),
                    cx.ty_path(
                        path_global(~[cx.ident_of("std"),
                                      cx.ident_of("pipes"),
                                      cx.ident_of(dir.to_str() + "Packet")],
                             dummy_sp())
                        .add_ty(cx.ty_path(
                            path(~[cx.ident_of("super"),
                                   self.data_name()],
                                 dummy_sp())
                            .add_tys(cx.ty_vars(
                                &self.generics.ty_params)), @opt_vec::Empty)),
                        @opt_vec::Empty),
                    cx.strip_bounds(&self.generics)));
        }
        else {
            items.push(
                cx.item_ty_poly(
                    self.span,
                    self.data_name(),
                    cx.ty_path(
                        path_global(~[cx.ident_of("std"),
                                      cx.ident_of("pipes"),
                                      cx.ident_of(dir.to_str()
                                                  + "PacketBuffered")],
                             dummy_sp())
                        .add_tys(~[cx.ty_path(
                            path(~[cx.ident_of("super"),
                                   self.data_name()],
                                        dummy_sp())
                            .add_tys(cx.ty_vars_global(
                                &self.generics.ty_params)), @opt_vec::Empty),
                                   self.proto.buffer_ty_path(cx)]), @opt_vec::Empty),
                    cx.strip_bounds(&self.generics)));
        };
        items
    }
}

impl gen_init for protocol {
    fn gen_init(&self, cx: @ExtCtxt) -> @ast::item {
        let ext_cx = cx;

        debug!("gen_init");
        let start_state = self.states[0];

        let body = if !self.is_bounded() {
            quote_expr!( ::std::pipes::entangle() )
        }
        else {
            self.gen_init_bounded(ext_cx)
        };

        cx.parse_item(fmt!("pub fn init%s() -> (server::%s, client::%s)\
                            { pub use std::pipes::HasBuffer; %s }",
                           start_state.generics.to_source(),
                           start_state.to_ty(cx).to_source(),
                           start_state.to_ty(cx).to_source(),
                           body.to_source()).to_managed())
    }

    fn gen_buffer_init(&self, ext_cx: @ExtCtxt) -> @ast::expr {
        ext_cx.expr_struct(
            dummy_sp(),
            path(~[ext_cx.ident_of("__Buffer")],
                 dummy_sp()),
            self.states.iter().transform(|s| {
                let fty = s.to_ty(ext_cx);
                ext_cx.field_imm(dummy_sp(),
                                 ext_cx.ident_of(s.name),
                                 quote_expr!(
                                     ::std::pipes::mk_packet::<$fty>()
                                 ))
            }).collect())
    }

    fn gen_init_bounded(&self, ext_cx: @ExtCtxt) -> @ast::expr {
        debug!("gen_init_bounded");
        let buffer_fields = self.gen_buffer_init(ext_cx);
        let buffer = quote_expr!(~::std::pipes::Buffer {
            header: ::std::pipes::BufferHeader(),
            data: $buffer_fields,
        });

        let entangle_body = ext_cx.expr_blk(
            ext_cx.blk(
                dummy_sp(),
                self.states.iter().transform(
                    |s| ext_cx.parse_stmt(
                        fmt!("data.%s.set_buffer(buffer)",
                             s.name).to_managed())).collect(),
                Some(ext_cx.parse_expr(fmt!(
                    "::std::ptr::to_mut_unsafe_ptr(&mut (data.%s))",
                    self.states[0].name).to_managed()))));

        quote_expr!({
            let buffer = $buffer;
            do ::std::pipes::entangle_buffer(buffer) |buffer, data| {
                $entangle_body
            }
        })
    }

    fn buffer_ty_path(&self, cx: @ExtCtxt) -> @ast::Ty {
        let mut params: OptVec<ast::TyParam> = opt_vec::Empty;
        for (copy self.states).iter().advance |s| {
            for s.generics.ty_params.each |tp| {
                match params.iter().find_(|tpp| tp.ident == tpp.ident) {
                  None => params.push(*tp),
                  _ => ()
                }
            }
        }

        cx.ty_path(path(~[cx.ident_of("super"),
                          cx.ident_of("__Buffer")],
                        copy self.span)
                   .add_tys(cx.ty_vars_global(&params)), @opt_vec::Empty)
    }

    fn gen_buffer_type(&self, cx: @ExtCtxt) -> @ast::item {
        let ext_cx = cx;
        let mut params: OptVec<ast::TyParam> = opt_vec::Empty;
        let fields = do (copy self.states).iter().transform |s| {
            for s.generics.ty_params.each |tp| {
                match params.iter().find_(|tpp| tp.ident == tpp.ident) {
                  None => params.push(*tp),
                  _ => ()
                }
            }

            let ty = s.to_ty(cx);
            let fty = quote_ty!( ::std::pipes::Packet<$ty> );

            @spanned {
                node: ast::struct_field_ {
                    kind: ast::named_field(cx.ident_of(s.name),
                                           ast::inherited),
                    id: cx.next_id(),
                    ty: fty,
                    attrs: ~[],
                },
                span: dummy_sp()
            }
        }.collect();

        let generics = Generics {
            lifetimes: opt_vec::Empty,
            ty_params: params
        };

        cx.item_struct_poly(
            dummy_sp(),
            cx.ident_of("__Buffer"),
            ast::struct_def {
                fields: fields,
                ctor_id: None
            },
            cx.strip_bounds(&generics))
    }

    fn compile(&self, cx: @ExtCtxt) -> @ast::item {
        let mut items = ~[self.gen_init(cx)];
        let mut client_states = ~[];
        let mut server_states = ~[];

        for (copy self.states).iter().advance |s| {
            items += s.to_type_decls(cx);

            client_states += s.to_endpoint_decls(cx, send);
            server_states += s.to_endpoint_decls(cx, recv);
        }

        if self.is_bounded() {
            items.push(self.gen_buffer_type(cx))
        }

        items.push(cx.item_mod(copy self.span,
                               cx.ident_of("client"),
                               ~[], ~[],
                               client_states));
        items.push(cx.item_mod(copy self.span,
                               cx.ident_of("server"),
                               ~[], ~[],
                               server_states));

        // XXX: Would be nice if our generated code didn't violate
        // Rust coding conventions
        let allows = cx.attribute(
            copy self.span,
            cx.meta_list(copy self.span,
                         @"allow",
                         ~[cx.meta_word(copy self.span, @"non_camel_case_types"),
                           cx.meta_word(copy self.span, @"unused_mut")]));
        cx.item_mod(copy self.span, cx.ident_of(copy self.name),
                    ~[allows], ~[], items)
    }
}
