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
use syntax::fold::ast_fold;
use syntax::fold;
use syntax::opt_vec;

static STD_VERSION: &'static str = "0.9-pre";

pub fn maybe_inject_libstd_ref(sess: Session, crate: @ast::Crate)
                               -> @ast::Crate {
    if use_std(crate) {
        inject_libstd_ref(sess, crate)
    } else {
        crate
    }
}

fn use_std(crate: &ast::Crate) -> bool {
    !attr::contains_name(crate.attrs, "no_std")
}

fn no_prelude(attrs: &[ast::Attribute]) -> bool {
    attr::contains_name(attrs, "no_implicit_prelude")
}

fn spanned<T>(x: T) -> codemap::Spanned<T> {
    codemap::Spanned {
        node: x,
        span: dummy_sp(),
    }
}

struct StandardLibraryInjector {
    sess: Session,
}

impl fold::ast_fold for StandardLibraryInjector {
    fn fold_crate(&self, crate: &ast::Crate) -> ast::Crate {
        let version = STD_VERSION.to_managed();
        let vi1 = ast::view_item {
            node: ast::view_item_extern_mod(self.sess.ident_of("std"),
                                            None,
                                            ~[],
                                            ast::DUMMY_NODE_ID),
            attrs: ~[
                attr::mk_attr(attr::mk_name_value_item_str(@"vers", version))
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
            new_module = self.fold_mod(&new_module);
        }

        // FIXME #2543: Bad copy.
        ast::Crate {
            module: new_module,
            ..(*crate).clone()
        }
    }

    fn fold_item(&self, item: @ast::item) -> Option<@ast::item> {
        if !no_prelude(item.attrs) {
            // only recur if there wasn't `#[no_implicit_prelude];`
            // on this item, i.e. this means that the prelude is not
            // implicitly imported though the whole subtree
            fold::noop_fold_item(item, self)
        } else {
            Some(item)
        }
    }

    fn fold_mod(&self, module: &ast::_mod) -> ast::_mod {
        let prelude_path = ast::Path {
            span: dummy_sp(),
            global: false,
            segments: ~[
                ast::PathSegment {
                    identifier: self.sess.ident_of("std"),
                    lifetime: None,
                    types: opt_vec::Empty,
                },
                ast::PathSegment {
                    identifier: self.sess.ident_of("prelude"),
                    lifetime: None,
                    types: opt_vec::Empty,
                },
            ],
        };

        let vp = @spanned(ast::view_path_glob(prelude_path,
                                              ast::DUMMY_NODE_ID));
        let vi2 = ast::view_item {
            node: ast::view_item_use(~[vp]),
            attrs: ~[],
            vis: ast::private,
            span: dummy_sp(),
        };

        let vis = vec::append(~[vi2], module.view_items);

        // FIXME #2543: Bad copy.
        let new_module = ast::_mod {
            view_items: vis,
            ..(*module).clone()
        };
        fold::noop_fold_mod(&new_module, self)
    }
}

fn inject_libstd_ref(sess: Session, crate: &ast::Crate) -> @ast::Crate {
    let fold = StandardLibraryInjector {
        sess: sess,
    };
    @fold.fold_crate(crate)
}
