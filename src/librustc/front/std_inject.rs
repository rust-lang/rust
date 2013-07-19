// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.


use driver::session::Session;

use std::vec;
use syntax::ast;
use syntax::attr;
use syntax::codemap::dummy_sp;
use syntax::codemap;
use syntax::fold;

static STD_VERSION: &'static str = "0.8-pre";

pub fn maybe_inject_libstd_ref(sess: Session, crate: @ast::crate)
                               -> @ast::crate {
    if use_std(crate) {
        inject_libstd_ref(sess, crate)
    } else {
        crate
    }
}

fn use_std(crate: &ast::crate) -> bool {
    !attr::attrs_contains_name(crate.node.attrs, "no_std")
}
fn no_prelude(attrs: &[ast::attribute]) -> bool {
    attr::attrs_contains_name(attrs, "no_implicit_prelude")
}

fn inject_libstd_ref(sess: Session, crate: &ast::crate) -> @ast::crate {
    fn spanned<T>(x: T) -> codemap::spanned<T> {
        codemap::spanned { node: x, span: dummy_sp() }
    }

    let precursor = @fold::AstFoldFns {
        fold_crate: |crate, span, fld| {
            let n1 = sess.next_node_id();
            let vi1 = ast::view_item {
                node: ast::view_item_extern_mod(
                        sess.ident_of("std"), ~[], n1),
                attrs: ~[
                    spanned(ast::attribute_ {
                        style: ast::attr_inner,
                        value: @spanned(ast::meta_name_value(
                            @"vers",
                            spanned(ast::lit_str(STD_VERSION.to_managed()))
                        )),
                        is_sugared_doc: false
                    })
                ],
                vis: ast::private,
                span: dummy_sp()
            };

            let vis = vec::append(~[vi1], crate.module.view_items);
            let mut new_module = ast::_mod {
                view_items: vis,
                ..crate.module.clone()
            };

            if !no_prelude(crate.attrs) {
                // only add `use std::prelude::*;` if there wasn't a
                // `#[no_implicit_prelude];` at the crate level.
                new_module = fld.fold_mod(&new_module);
            }

            // FIXME #2543: Bad copy.
            let new_crate = ast::crate_ {
                module: new_module,
                ..(*crate).clone()
            };
            (new_crate, span)
        },
        fold_item: |item, fld| {
            if !no_prelude(item.attrs) {
                // only recur if there wasn't `#[no_implicit_prelude];`
                // on this item, i.e. this means that the prelude is not
                // implicitly imported though the whole subtree
                fold::noop_fold_item(item, fld)
            } else {
                Some(item)
            }
        },
        fold_mod: |module, fld| {
            let n2 = sess.next_node_id();

            let prelude_path = ast::Path {
                span: dummy_sp(),
                global: false,
                idents: ~[
                    sess.ident_of("std"),
                    sess.ident_of("prelude")
                ],
                rp: None,
                types: ~[]
            };

            let vp = @spanned(ast::view_path_glob(prelude_path, n2));
            let vi2 = ast::view_item { node: ast::view_item_use(~[vp]),
                                        attrs: ~[],
                                        vis: ast::private,
                                        span: dummy_sp() };

            let vis = vec::append(~[vi2], module.view_items);

            // FIXME #2543: Bad copy.
            let new_module = ast::_mod {
                view_items: vis,
                ..(*module).clone()
            };
            fold::noop_fold_mod(&new_module, fld)
        },
        ..*fold::default_ast_fold()
    };

    let fold = fold::make_fold(precursor);
    @fold.fold_crate(crate)
}
