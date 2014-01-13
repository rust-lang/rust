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
use syntax::codemap::DUMMY_SP;
use syntax::codemap;
use syntax::fold::Folder;
use syntax::fold;
use syntax::opt_vec;
use syntax::util::small_vector::SmallVector;

pub fn maybe_inject_libstd_ref(sess: Session, crate: ast::Crate)
                               -> ast::Crate {
    if use_std(&crate) {
        inject_libstd_ref(sess, crate)
    } else {
        crate
    }
}

fn use_std(crate: &ast::Crate) -> bool {
    !attr::contains_name(crate.attrs, "no_std")
}

fn use_uv(crate: &ast::Crate) -> bool {
    !attr::contains_name(crate.attrs, "no_uv")
}

fn no_prelude(attrs: &[ast::Attribute]) -> bool {
    attr::contains_name(attrs, "no_implicit_prelude")
}

fn spanned<T>(x: T) -> codemap::Spanned<T> {
    codemap::Spanned {
        node: x,
        span: DUMMY_SP,
    }
}

struct StandardLibraryInjector {
    sess: Session,
}

impl fold::Folder for StandardLibraryInjector {
    fn fold_crate(&mut self, crate: ast::Crate) -> ast::Crate {
        let mut vis = ~[ast::ViewItem {
            node: ast::ViewItemExternMod(self.sess.ident_of("std"),
                                         None,
                                         ast::DUMMY_NODE_ID),
            attrs: ~[],
            vis: ast::Private,
            span: DUMMY_SP
        }];

        if use_uv(&crate) && !self.sess.building_library.get() {
            vis.push(ast::ViewItem {
                node: ast::ViewItemExternMod(self.sess.ident_of("green"),
                                             None,
                                             ast::DUMMY_NODE_ID),
                attrs: ~[],
                vis: ast::Private,
                span: DUMMY_SP
            });
            vis.push(ast::ViewItem {
                node: ast::ViewItemExternMod(self.sess.ident_of("rustuv"),
                                             None,
                                             ast::DUMMY_NODE_ID),
                attrs: ~[],
                vis: ast::Private,
                span: DUMMY_SP
            });
        }

        vis.push_all(crate.module.view_items);
        let mut new_module = ast::Mod {
            view_items: vis,
            ..crate.module.clone()
        };

        if !no_prelude(crate.attrs) {
            // only add `use std::prelude::*;` if there wasn't a
            // `#[no_implicit_prelude];` at the crate level.
            new_module = self.fold_mod(&new_module);
        }

        ast::Crate {
            module: new_module,
            ..crate
        }
    }

    fn fold_item(&mut self, item: @ast::Item) -> SmallVector<@ast::Item> {
        if !no_prelude(item.attrs) {
            // only recur if there wasn't `#[no_implicit_prelude];`
            // on this item, i.e. this means that the prelude is not
            // implicitly imported though the whole subtree
            fold::noop_fold_item(item, self)
        } else {
            SmallVector::one(item)
        }
    }

    fn fold_mod(&mut self, module: &ast::Mod) -> ast::Mod {
        let prelude_path = ast::Path {
            span: DUMMY_SP,
            global: false,
            segments: ~[
                ast::PathSegment {
                    identifier: self.sess.ident_of("std"),
                    lifetimes: opt_vec::Empty,
                    types: opt_vec::Empty,
                },
                ast::PathSegment {
                    identifier: self.sess.ident_of("prelude"),
                    lifetimes: opt_vec::Empty,
                    types: opt_vec::Empty,
                },
            ],
        };

        let vp = @spanned(ast::ViewPathGlob(prelude_path, ast::DUMMY_NODE_ID));
        let vi2 = ast::ViewItem {
            node: ast::ViewItemUse(~[vp]),
            attrs: ~[],
            vis: ast::Private,
            span: DUMMY_SP,
        };

        let vis = vec::append(~[vi2], module.view_items);

        // FIXME #2543: Bad copy.
        let new_module = ast::Mod {
            view_items: vis,
            ..(*module).clone()
        };
        fold::noop_fold_mod(&new_module, self)
    }
}

fn inject_libstd_ref(sess: Session, crate: ast::Crate) -> ast::Crate {
    let mut fold = StandardLibraryInjector {
        sess: sess,
    };
    fold.fold_crate(crate)
}
