// Copyright 2016 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use rustc::middle::allocator::AllocatorKind;
use rustc_errors;
use syntax::abi::Abi;
use syntax::ast::{Crate, Attribute, LitKind, StrStyle, ExprKind};
use syntax::ast::{Unsafety, Constness, Generics, Mutability, Ty, Mac, Arg};
use syntax::ast::{self, Ident, Item, ItemKind, TyKind, Visibility, Expr};
use syntax::attr;
use syntax::codemap::dummy_spanned;
use syntax::codemap::{ExpnInfo, NameAndSpan, MacroAttribute};
use syntax::ext::base::ExtCtxt;
use syntax::ext::base::Resolver;
use syntax::ext::build::AstBuilder;
use syntax::ext::expand::ExpansionConfig;
use syntax::ext::hygiene::{Mark, SyntaxContext};
use syntax::fold::{self, Folder};
use syntax::parse::ParseSess;
use syntax::ptr::P;
use syntax::symbol::Symbol;
use syntax::util::small_vector::SmallVector;
use syntax_pos::{Span, DUMMY_SP};

use {AllocatorMethod, AllocatorTy, ALLOCATOR_METHODS};

pub fn modify(sess: &ParseSess,
              resolver: &mut Resolver,
              krate: Crate,
              handler: &rustc_errors::Handler) -> ast::Crate {
    ExpandAllocatorDirectives {
        handler,
        sess,
        resolver,
        found: false,
    }.fold_crate(krate)
}

struct ExpandAllocatorDirectives<'a> {
    found: bool,
    handler: &'a rustc_errors::Handler,
    sess: &'a ParseSess,
    resolver: &'a mut Resolver,
}

impl<'a> Folder for ExpandAllocatorDirectives<'a> {
    fn fold_item(&mut self, item: P<Item>) -> SmallVector<P<Item>> {
        let name = if attr::contains_name(&item.attrs, "global_allocator") {
            "global_allocator"
        } else {
            return fold::noop_fold_item(item, self)
        };
        match item.node {
            ItemKind::Static(..) => {}
            _ => {
                self.handler.span_err(item.span, "allocators must be statics");
                return SmallVector::one(item)
            }
        }

        if self.found {
            self.handler.span_err(item.span, "cannot define more than one \
                                              #[global_allocator]");
            return SmallVector::one(item)
        }
        self.found = true;

        let mark = Mark::fresh(Mark::root());
        mark.set_expn_info(ExpnInfo {
            call_site: DUMMY_SP,
            callee: NameAndSpan {
                format: MacroAttribute(Symbol::intern(name)),
                span: None,
                allow_internal_unstable: true,
                allow_internal_unsafe: false,
            }
        });
        let span = item.span.with_ctxt(SyntaxContext::empty().apply_mark(mark));
        let ecfg = ExpansionConfig::default(name.to_string());
        let mut f = AllocFnFactory {
            span,
            kind: AllocatorKind::Global,
            global: item.ident,
            alloc: Ident::from_str("alloc"),
            cx: ExtCtxt::new(self.sess, ecfg, self.resolver),
        };
        let super_path = f.cx.path(f.span, vec![
            Ident::from_str("super"),
            f.global,
        ]);
        let mut items = vec![
            f.cx.item_extern_crate(f.span, f.alloc),
            f.cx.item_use_simple(f.span, Visibility::Inherited, super_path),
        ];
        for method in ALLOCATOR_METHODS {
            items.push(f.allocator_fn(method));
        }
        let name = f.kind.fn_name("allocator_abi");
        let allocator_abi = Ident::with_empty_ctxt(Symbol::gensym(&name));
        let module = f.cx.item_mod(span, span, allocator_abi, Vec::new(), items);
        let module = f.cx.monotonic_expander().fold_item(module).pop().unwrap();

        let mut ret = SmallVector::new();
        ret.push(item);
        ret.push(module);
        return ret
    }

    fn fold_mac(&mut self, mac: Mac) -> Mac {
        fold::noop_fold_mac(mac, self)
    }
}

struct AllocFnFactory<'a> {
    span: Span,
    kind: AllocatorKind,
    global: Ident,
    alloc: Ident,
    cx: ExtCtxt<'a>,
}

impl<'a> AllocFnFactory<'a> {
    fn allocator_fn(&self, method: &AllocatorMethod) -> P<Item> {
        let mut abi_args = Vec::new();
        let mut i = 0;
        let ref mut mk = || {
            let name = Ident::from_str(&format!("arg{}", i));
            i += 1;
            name
        };
        let args = method.inputs.iter().map(|ty| {
            self.arg_ty(ty, &mut abi_args, mk)
        }).collect();
        let result = self.call_allocator(method.name, args);
        let (output_ty, output_expr) =
            self.ret_ty(&method.output, &mut abi_args, mk, result);
        let kind = ItemKind::Fn(self.cx.fn_decl(abi_args, output_ty),
                                Unsafety::Unsafe,
                                dummy_spanned(Constness::NotConst),
                                Abi::Rust,
                                Generics::default(),
                                self.cx.block_expr(output_expr));
        self.cx.item(self.span,
                     Ident::from_str(&self.kind.fn_name(method.name)),
                     self.attrs(),
                     kind)
    }

    fn call_allocator(&self, method: &str, mut args: Vec<P<Expr>>) -> P<Expr> {
        let method = self.cx.path(self.span, vec![
            self.alloc,
            Ident::from_str("heap"),
            Ident::from_str("Alloc"),
            Ident::from_str(method),
        ]);
        let method = self.cx.expr_path(method);
        let allocator = self.cx.path_ident(self.span, self.global);
        let allocator = self.cx.expr_path(allocator);
        let allocator = self.cx.expr_addr_of(self.span, allocator);
        let allocator = self.cx.expr_mut_addr_of(self.span, allocator);
        args.insert(0, allocator);

        self.cx.expr_call(self.span, method, args)
    }

    fn attrs(&self) -> Vec<Attribute> {
        let key = Symbol::intern("linkage");
        let value = LitKind::Str(Symbol::intern("external"), StrStyle::Cooked);
        let linkage = self.cx.meta_name_value(self.span, key, value);

        let no_mangle = Symbol::intern("no_mangle");
        let no_mangle = self.cx.meta_word(self.span, no_mangle);

        let special = Symbol::intern("rustc_std_internal_symbol");
        let special = self.cx.meta_word(self.span, special);
        vec![
            self.cx.attribute(self.span, linkage),
            self.cx.attribute(self.span, no_mangle),
            self.cx.attribute(self.span, special),
        ]
    }

    fn arg_ty(&self,
              ty: &AllocatorTy,
              args: &mut Vec<Arg>,
              ident: &mut FnMut() -> Ident) -> P<Expr> {
        match *ty {
            AllocatorTy::Layout => {
                let usize = self.cx.path_ident(self.span, Ident::from_str("usize"));
                let ty_usize = self.cx.ty_path(usize);
                let size = ident();
                let align = ident();
                args.push(self.cx.arg(self.span, size, ty_usize.clone()));
                args.push(self.cx.arg(self.span, align, ty_usize));

                let layout_new = self.cx.path(self.span, vec![
                    self.alloc,
                    Ident::from_str("heap"),
                    Ident::from_str("Layout"),
                    Ident::from_str("from_size_align_unchecked"),
                ]);
                let layout_new = self.cx.expr_path(layout_new);
                let size = self.cx.expr_ident(self.span, size);
                let align = self.cx.expr_ident(self.span, align);
                let layout = self.cx.expr_call(self.span,
                                               layout_new,
                                               vec![size, align]);
                layout
            }

            AllocatorTy::LayoutRef => {
                let ident = ident();
                args.push(self.cx.arg(self.span, ident, self.ptr_u8()));

                // Convert our `arg: *const u8` via:
                //
                //      &*(arg as *const Layout)
                let expr = self.cx.expr_ident(self.span, ident);
                let expr = self.cx.expr_cast(self.span, expr, self.layout_ptr());
                let expr = self.cx.expr_deref(self.span, expr);
                self.cx.expr_addr_of(self.span, expr)
            }

            AllocatorTy::AllocErr => {
                // We're creating:
                //
                //      (*(arg as *const AllocErr)).clone()
                let ident = ident();
                args.push(self.cx.arg(self.span, ident, self.ptr_u8()));
                let expr = self.cx.expr_ident(self.span, ident);
                let expr = self.cx.expr_cast(self.span, expr, self.alloc_err_ptr());
                let expr = self.cx.expr_deref(self.span, expr);
                self.cx.expr_method_call(
                    self.span,
                    expr,
                    Ident::from_str("clone"),
                    Vec::new()
                )
            }

            AllocatorTy::Ptr => {
                let ident = ident();
                args.push(self.cx.arg(self.span, ident, self.ptr_u8()));
                self.cx.expr_ident(self.span, ident)
            }

            AllocatorTy::ResultPtr |
            AllocatorTy::ResultExcess |
            AllocatorTy::ResultUnit |
            AllocatorTy::Bang |
            AllocatorTy::UsizePair |
            AllocatorTy::Unit => {
                panic!("can't convert AllocatorTy to an argument")
            }
        }
    }

    fn ret_ty(&self,
              ty: &AllocatorTy,
              args: &mut Vec<Arg>,
              ident: &mut FnMut() -> Ident,
              expr: P<Expr>) -> (P<Ty>, P<Expr>)
    {
        match *ty {
            AllocatorTy::UsizePair => {
                // We're creating:
                //
                //      let arg = #expr;
                //      *min = arg.0;
                //      *max = arg.1;

                let min = ident();
                let max = ident();

                args.push(self.cx.arg(self.span, min, self.ptr_usize()));
                args.push(self.cx.arg(self.span, max, self.ptr_usize()));

                let ident = ident();
                let stmt = self.cx.stmt_let(self.span, false, ident, expr);
                let min = self.cx.expr_ident(self.span, min);
                let max = self.cx.expr_ident(self.span, max);
                let layout = self.cx.expr_ident(self.span, ident);
                let assign_min = self.cx.expr(self.span, ExprKind::Assign(
                    self.cx.expr_deref(self.span, min),
                    self.cx.expr_tup_field_access(self.span, layout.clone(), 0),
                ));
                let assign_min = self.cx.stmt_semi(assign_min);
                let assign_max = self.cx.expr(self.span, ExprKind::Assign(
                    self.cx.expr_deref(self.span, max),
                    self.cx.expr_tup_field_access(self.span, layout.clone(), 1),
                ));
                let assign_max = self.cx.stmt_semi(assign_max);

                let stmts = vec![stmt, assign_min, assign_max];
                let block = self.cx.block(self.span, stmts);
                let ty_unit = self.cx.ty(self.span, TyKind::Tup(Vec::new()));
                (ty_unit, self.cx.expr_block(block))
            }

            AllocatorTy::ResultExcess => {
                // We're creating:
                //
                //      match #expr {
                //          Ok(ptr) => {
                //              *excess = ptr.1;
                //              ptr.0
                //          }
                //          Err(e) => {
                //              ptr::write(err_ptr, e);
                //              0 as *mut u8
                //          }
                //      }

                let excess_ptr = ident();
                args.push(self.cx.arg(self.span, excess_ptr, self.ptr_usize()));
                let excess_ptr = self.cx.expr_ident(self.span, excess_ptr);

                let err_ptr = ident();
                args.push(self.cx.arg(self.span, err_ptr, self.ptr_u8()));
                let err_ptr = self.cx.expr_ident(self.span, err_ptr);
                let err_ptr = self.cx.expr_cast(self.span,
                                                err_ptr,
                                                self.alloc_err_ptr());

                let name = ident();
                let ok_expr = {
                    let ptr = self.cx.expr_ident(self.span, name);
                    let write = self.cx.expr(self.span, ExprKind::Assign(
                        self.cx.expr_deref(self.span, excess_ptr),
                        self.cx.expr_tup_field_access(self.span, ptr.clone(), 1),
                    ));
                    let write = self.cx.stmt_semi(write);
                    let ret = self.cx.expr_tup_field_access(self.span,
                                                            ptr.clone(),
                                                            0);
                    let ret = self.cx.stmt_expr(ret);
                    let block = self.cx.block(self.span, vec![write, ret]);
                    self.cx.expr_block(block)
                };
                let pat = self.cx.pat_ident(self.span, name);
                let ok = self.cx.path_ident(self.span, Ident::from_str("Ok"));
                let ok = self.cx.pat_tuple_struct(self.span, ok, vec![pat]);
                let ok = self.cx.arm(self.span, vec![ok], ok_expr);

                let name = ident();
                let err_expr = {
                    let err = self.cx.expr_ident(self.span, name);
                    let write = self.cx.path(self.span, vec![
                        self.alloc,
                        Ident::from_str("heap"),
                        Ident::from_str("__core"),
                        Ident::from_str("ptr"),
                        Ident::from_str("write"),
                    ]);
                    let write = self.cx.expr_path(write);
                    let write = self.cx.expr_call(self.span, write,
                                                  vec![err_ptr, err]);
                    let write = self.cx.stmt_semi(write);
                    let null = self.cx.expr_usize(self.span, 0);
                    let null = self.cx.expr_cast(self.span, null, self.ptr_u8());
                    let null = self.cx.stmt_expr(null);
                    let block = self.cx.block(self.span, vec![write, null]);
                    self.cx.expr_block(block)
                };
                let pat = self.cx.pat_ident(self.span, name);
                let err = self.cx.path_ident(self.span, Ident::from_str("Err"));
                let err = self.cx.pat_tuple_struct(self.span, err, vec![pat]);
                let err = self.cx.arm(self.span, vec![err], err_expr);

                let expr = self.cx.expr_match(self.span, expr, vec![ok, err]);
                (self.ptr_u8(), expr)
            }

            AllocatorTy::ResultPtr => {
                // We're creating:
                //
                //      match #expr {
                //          Ok(ptr) => ptr,
                //          Err(e) => {
                //              ptr::write(err_ptr, e);
                //              0 as *mut u8
                //          }
                //      }

                let err_ptr = ident();
                args.push(self.cx.arg(self.span, err_ptr, self.ptr_u8()));
                let err_ptr = self.cx.expr_ident(self.span, err_ptr);
                let err_ptr = self.cx.expr_cast(self.span,
                                                err_ptr,
                                                self.alloc_err_ptr());

                let name = ident();
                let ok_expr = self.cx.expr_ident(self.span, name);
                let pat = self.cx.pat_ident(self.span, name);
                let ok = self.cx.path_ident(self.span, Ident::from_str("Ok"));
                let ok = self.cx.pat_tuple_struct(self.span, ok, vec![pat]);
                let ok = self.cx.arm(self.span, vec![ok], ok_expr);

                let name = ident();
                let err_expr = {
                    let err = self.cx.expr_ident(self.span, name);
                    let write = self.cx.path(self.span, vec![
                        self.alloc,
                        Ident::from_str("heap"),
                        Ident::from_str("__core"),
                        Ident::from_str("ptr"),
                        Ident::from_str("write"),
                    ]);
                    let write = self.cx.expr_path(write);
                    let write = self.cx.expr_call(self.span, write,
                                                  vec![err_ptr, err]);
                    let write = self.cx.stmt_semi(write);
                    let null = self.cx.expr_usize(self.span, 0);
                    let null = self.cx.expr_cast(self.span, null, self.ptr_u8());
                    let null = self.cx.stmt_expr(null);
                    let block = self.cx.block(self.span, vec![write, null]);
                    self.cx.expr_block(block)
                };
                let pat = self.cx.pat_ident(self.span, name);
                let err = self.cx.path_ident(self.span, Ident::from_str("Err"));
                let err = self.cx.pat_tuple_struct(self.span, err, vec![pat]);
                let err = self.cx.arm(self.span, vec![err], err_expr);

                let expr = self.cx.expr_match(self.span, expr, vec![ok, err]);
                (self.ptr_u8(), expr)
            }

            AllocatorTy::ResultUnit => {
                // We're creating:
                //
                //      #expr.is_ok() as u8

                let cast = self.cx.expr_method_call(
                    self.span,
                    expr,
                    Ident::from_str("is_ok"),
                    Vec::new()
                );
                let u8 = self.cx.path_ident(self.span, Ident::from_str("u8"));
                let u8 = self.cx.ty_path(u8);
                let cast = self.cx.expr_cast(self.span, cast, u8.clone());
                (u8, cast)
            }

            AllocatorTy::Bang => {
                (self.cx.ty(self.span, TyKind::Never), expr)
            }

            AllocatorTy::Unit => {
                (self.cx.ty(self.span, TyKind::Tup(Vec::new())), expr)
            }

            AllocatorTy::AllocErr |
            AllocatorTy::Layout |
            AllocatorTy::LayoutRef |
            AllocatorTy::Ptr => {
                panic!("can't convert AllocatorTy to an output")
            }
        }
    }

    fn ptr_u8(&self) -> P<Ty> {
        let u8 = self.cx.path_ident(self.span, Ident::from_str("u8"));
        let ty_u8 = self.cx.ty_path(u8);
        self.cx.ty_ptr(self.span, ty_u8, Mutability::Mutable)
    }

    fn ptr_usize(&self) -> P<Ty> {
        let usize = self.cx.path_ident(self.span, Ident::from_str("usize"));
        let ty_usize = self.cx.ty_path(usize);
        self.cx.ty_ptr(self.span, ty_usize, Mutability::Mutable)
    }

    fn layout_ptr(&self) -> P<Ty> {
        let layout = self.cx.path(self.span, vec![
            self.alloc,
            Ident::from_str("heap"),
            Ident::from_str("Layout"),
        ]);
        let layout = self.cx.ty_path(layout);
        self.cx.ty_ptr(self.span, layout, Mutability::Mutable)
    }

    fn alloc_err_ptr(&self) -> P<Ty> {
        let err = self.cx.path(self.span, vec![
            self.alloc,
            Ident::from_str("heap"),
            Ident::from_str("AllocErr"),
        ]);
        let err = self.cx.ty_path(err);
        self.cx.ty_ptr(self.span, err, Mutability::Mutable)
    }
}
