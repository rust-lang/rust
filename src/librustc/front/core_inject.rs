// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use core::prelude::*;

use driver::session::Session;

use core::vec;
use syntax::ast;
use syntax::ast_util::*;
use syntax::attr;
use syntax::codemap;
use syntax::fold;

export maybe_inject_libcore_ref;

const CORE_VERSION: &static/str = "0.6";

fn maybe_inject_libcore_ref(sess: Session,
                            crate: @ast::crate) -> @ast::crate {
    if use_core(crate) {
        inject_libcore_ref(sess, crate)
    } else {
        crate
    }
}

fn use_core(crate: @ast::crate) -> bool {
    !attr::attrs_contains_name(crate.node.attrs, ~"no_core")
}

fn inject_libcore_ref(sess: Session,
                      crate: @ast::crate) -> @ast::crate {
    fn spanned<T: Copy>(x: T) -> ast::spanned<T> {
        return {node: x, span: dummy_sp()};
    }

    let precursor = @{
        fold_crate: |crate, span, fld| {
            let n1 = sess.next_node_id();
            let vi1 = @{node: ast::view_item_use(sess.ident_of(~"core"),
                                                 ~[],
                                                 n1),
                        attrs: ~[
                            spanned({
                                style: ast::attr_inner,
                                value: spanned(ast::meta_name_value(
                                    ~"vers",
                                    spanned(ast::lit_str(
                                        @CORE_VERSION.to_str()))
                                )),
                                is_sugared_doc: false
                            })
                        ],
                        vis: ast::private,
                        span: dummy_sp()};

            let vis = vec::append(~[vi1], crate.module.view_items);
            let mut new_module = {
                view_items: vis,
                ../*bad*/copy crate.module
            };
            new_module = fld.fold_mod(new_module);

            // XXX: Bad copy.
            let new_crate = { module: new_module, ..copy crate };
            (new_crate, span)
        },
        fold_mod: |module, fld| {
            let n2 = sess.next_node_id();

            let prelude_path = @{
                span: dummy_sp(),
                global: false,
                idents: ~[
                    sess.ident_of(~"core"),
                    sess.ident_of(~"prelude")
                ],
                rp: None,
                types: ~[]
            };

            let vp = @spanned(ast::view_path_glob(prelude_path, n2));
            let vi2 = @{node: ast::view_item_import(~[vp]),
                        attrs: ~[],
                        vis: ast::private,
                        span: dummy_sp()};

            let vis = vec::append(~[vi2], module.view_items);

            // XXX: Bad copy.
            let new_module = { view_items: vis, ..copy module };
            fold::noop_fold_mod(new_module, fld)
        },
        ..*fold::default_ast_fold()
    };

    let fold = fold::make_fold(precursor);
    @fold.fold_crate(*crate)
}
