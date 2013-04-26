// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! Prune things that are private

use extract;
use syntax::ast;
use syntax::ast_map;
use astsrv;
use doc;
use fold::Fold;
use fold;
use pass::Pass;

use core::util;

pub fn mk_pass() -> Pass {
    Pass {
        name: ~"prune_private",
        f: run
    }
}

pub fn run(srv: astsrv::Srv, doc: doc::Doc) -> doc::Doc {
    // First strip private methods out of impls
    let fold = Fold {
        ctxt: srv.clone(),
        fold_impl: fold_impl,
        .. fold::default_any_fold(srv.clone())
    };
    let doc = (fold.fold_doc)(&fold, doc);

    // Then strip private items and empty impls
    let fold = Fold {
        ctxt: srv.clone(),
        fold_mod: fold_mod,
        .. fold::default_any_fold(srv)
    };
    let doc = (fold.fold_doc)(&fold, doc);

    return doc;
}

fn fold_impl(
    fold: &fold::Fold<astsrv::Srv>,
    doc: doc::ImplDoc
) -> doc::ImplDoc {
    let doc = fold::default_seq_fold_impl(fold, doc);

    do astsrv::exec(fold.ctxt.clone()) |ctxt| {
        match *ctxt.ast_map.get(&doc.item.id) {
            ast_map::node_item(item, _) => {
                match item.node {
                    ast::item_impl(_, None, _, ref methods) => {
                        // Associated impls have complex rules for method visibility
                        strip_priv_methods(copy doc, *methods, item.vis)
                    }
                    ast::item_impl(_, Some(_), _ ,_) => {
                        // Trait impls don't
                        copy doc
                    }
                    _ => fail!()
                }
            }
            _ => fail!()
        }
    }
}

fn strip_priv_methods(
    doc: doc::ImplDoc,
    methods: &[@ast::method],
    item_vis: ast::visibility
) -> doc::ImplDoc {
    let methods = do (&doc.methods).filtered |method| {
        let ast_method = do methods.find |m| {
            extract::to_str(m.ident) == method.name
        };
        assert!(ast_method.is_some());
        let ast_method = ast_method.unwrap();
        match ast_method.vis {
            ast::public => true,
            ast::private => false,
            ast::inherited => item_vis == ast::public
        }
    };

    doc::ImplDoc {
        methods: methods,
        .. doc
    }
}

fn fold_mod(
    fold: &fold::Fold<astsrv::Srv>,
    doc: doc::ModDoc
) -> doc::ModDoc {
    let doc = fold::default_any_fold_mod(fold, doc);

    doc::ModDoc {
        items: doc.items.filtered(|ItemTag| {
            match ItemTag {
                &doc::ImplTag(ref doc) => {
                    if doc.trait_types.is_empty() {
                        // This is an associated impl. We have already pruned the
                        // non-visible methods. If there are any left then
                        // retain the impl, otherwise throw it away
                        !doc.methods.is_empty()
                    } else {
                        // This is a trait implementation, make it visible
                        // NB: This is not quite right since this could be an impl
                        // of a private trait. We can't know that without running
                        // resolve though.
                        true
                    }
                }
                _ => {
                    is_visible(fold.ctxt.clone(), ItemTag.item())
                }
            }
        }),
        .. doc
    }
}

fn is_visible(srv: astsrv::Srv, doc: doc::ItemDoc) -> bool {
    let id = doc.id;

    do astsrv::exec(srv) |ctxt| {
        match *ctxt.ast_map.get(&id) {
            ast_map::node_item(item, _) => {
                match &item.node {
                    &ast::item_impl(*) => {
                        // Impls handled elsewhere
                        fail!()
                    }
                    _ => {
                        // Otherwise just look at the visibility
                        item.vis == ast::public
                    }
                }
            }
            _ => util::unreachable()
        }
    }
}


#[cfg(test)]
mod test {
    use astsrv;
    use doc;
    use extract;
    use tystr_pass;
    use prune_private_pass::run;
    use core::vec;

    fn mk_doc(source: ~str) -> doc::Doc {
        do astsrv::from_str(copy source) |srv| {
            let doc = extract::from_srv(srv.clone(), ~"");
            let doc = tystr_pass::run(srv.clone(), doc);
            run(srv.clone(), doc)
        }
    }

    #[test]
    fn should_prune_items_without_pub_modifier() {
        let doc = mk_doc(~"mod a { }");
        assert!(vec::is_empty(doc.cratemod().mods()));
    }

    #[test]
    fn should_not_prune_trait_impls() {
        // Impls are more complicated
        let doc = mk_doc(
            ~" \
              trait Foo { } \
              impl Foo for int { } \
              ");
        assert!(!doc.cratemod().impls().is_empty());
    }

    #[test]
    fn should_prune_associated_methods_without_vis_modifier_on_impls_without_vis_modifier() {
        let doc = mk_doc(
            ~"impl Foo {\
              pub fn bar() { }\
              fn baz() { }\
              }");
        assert!(doc.cratemod().impls()[0].methods.len() == 1);
    }

    #[test]
    fn should_prune_priv_associated_methods_on_impls_without_vis_modifier() {
        let doc = mk_doc(
            ~"impl Foo {\
              pub fn bar() { }\
              priv fn baz() { }\
              }");
        assert!(doc.cratemod().impls()[0].methods.len() == 1);
    }

    #[test]
    fn should_prune_priv_associated_methods_on_pub_impls() {
        let doc = mk_doc(
            ~"pub impl Foo {\
              fn bar() { }\
              priv fn baz() { }\
              }");
        assert!(doc.cratemod().impls()[0].methods.len() == 1);
    }

    #[test]
    fn should_prune_associated_methods_without_vis_modifier_on_priv_impls() {
        let doc = mk_doc(
            ~"priv impl Foo {\
              pub fn bar() { }\
              fn baz() { }\
              }");
        assert!(doc.cratemod().impls()[0].methods.len() == 1);
    }

    #[test]
    fn should_prune_priv_associated_methods_on_priv_impls() {
        let doc = mk_doc(
            ~"priv impl Foo {\
              pub fn bar() { }\
              priv fn baz() { }\
              }");
        assert!(doc.cratemod().impls()[0].methods.len() == 1);
    }

    #[test]
    fn should_prune_associated_impls_with_no_pub_methods() {
        let doc = mk_doc(
            ~"priv impl Foo {\
              fn baz() { }\
              }");
        assert!(doc.cratemod().impls().is_empty());
    }

    #[test]
    fn should_not_prune_associated_impls_with_pub_methods() {
        let doc = mk_doc(
            ~" \
              impl Foo { pub fn bar() { } } \
              ");
        assert!(!doc.cratemod().impls().is_empty());
    }
}
