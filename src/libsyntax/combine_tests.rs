// Copyright 2018 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use ast::*;
use ptr::P;
use fold;
use fold::Folder;
use symbol::keywords;
use codemap::dummy_spanned;
use syntax_pos::DUMMY_SP;
use syntax_pos::symbol::Symbol;
use ext::base::ExtCtxt;

pub struct RootPathFolder<'a, 'b: 'a> {
    pub cx: &'a mut ExtCtxt<'b>,
}

impl<'a, 'b> fold::Folder for RootPathFolder<'a, 'b> {
    fn fold_item_simple(&mut self, item: Item) -> Item {
        if item.ident.name == keywords::Invalid.name() {
            return fold::noop_fold_item_simple(item, self)
        }
        if let ItemKind::Mod(..) = item.node {
            make_crate_like_module(item, self.cx)
        } else {
            fold::noop_fold_item_simple(item, self)
        }
    }

    fn fold_mac(&mut self, mac: Mac) -> Mac {
        mac
    }
}

pub struct PathFolder {
    pub root: Ident,
}

impl PathFolder {
    fn fold_qpath(&mut self, qself: &mut Option<QSelf>, path: &mut Path) {
        let old = path.segments.len();
        *path = self.fold_path(path.clone());
        let add = path.segments.len() - old;
        qself.as_mut().map(|qself| {
            qself.position += add;
            qself.ty = self.fold_ty(qself.ty.clone());
        });
    }

    fn fold_absolute_path(&mut self, path: &mut Path) {
        let pos = {
            let get = |i| {
                path.segments.get(i).map(|p: &PathSegment| p.identifier.name)
            };
            if get(0) == Some(keywords::SelfValue.name()) ||
                get(0) == Some(keywords::DollarCrate.name()) ||
                get(0) == Some(keywords::Super.name()) {
                None
            } else {
                let mut i = 0;
                if get(i) == Some(keywords::CrateRoot.name()) {
                    i += 1;
                }
                if get(i) == Some(keywords::Crate.name()) {
                    i += 1;
                }
                Some(i)
            }
        };
        if let Some(pos) = pos {
            path.segments.insert(pos, PathSegment {
                identifier: self.root,
                span: path.span,
                parameters: None,
            });
        }
    }
}

impl fold::Folder for PathFolder {
    fn fold_use_tree(&mut self, mut use_tree: UseTree) -> UseTree {
        self.fold_absolute_path(&mut use_tree.prefix);
        use_tree
    }

    fn fold_vis(&mut self, mut vis: Visibility) -> Visibility {
        match vis.node {
            VisibilityKind::Restricted { ref mut path, .. } => self.fold_absolute_path(path),
            _ => (),
        }
        vis
    }

    fn fold_path(&mut self, mut p: Path) -> Path {
        if let Some(first) = p.segments.first().cloned() {
            if first.identifier.name == keywords::CrateRoot.name() {
                let idx = if p.segments.get(1).map(|p| p.identifier.name) ==
                                Some(keywords::Crate.name()) {
                    2
                } else {
                    1
                };
                p.segments.insert(idx, PathSegment {
                    identifier: self.root,
                    span: p.span,
                    parameters: None,
                });
            }
        }
        fold::noop_fold_path(p, self)
    }

    fn fold_ty(&mut self, mut t: P<Ty>) -> P<Ty> {
        if match t.node  {
            TyKind::Path(ref mut qself, ref mut path) => {
                self.fold_qpath(qself, path);
                true
            }
            _ => false,
        } {
            return t;
        }
        fold::noop_fold_ty(t, self)
    }

    fn fold_pat(&mut self, mut p: P<Pat>) -> P<Pat> {
        if match p.node  {
            PatKind::Path(ref mut qself, ref mut path) => {
                self.fold_qpath(qself, path);
                true
            }
            _ => false,
        } {
            return p;
        }
        fold::noop_fold_pat(p, self)
    }

    fn fold_expr(&mut self, mut e: P<Expr>) -> P<Expr> {
        if match e.node  {
            ExprKind::Path(ref mut qself, ref mut path) => {
                self.fold_qpath(qself, path);
                true
            }
            _ => false,
        } {
            return e;
        }
        e.map(|e| fold::noop_fold_expr(e, self))
    }

    fn fold_mac(&mut self, mut mac: Mac) -> Mac {
        mac.node.path = self.fold_path(mac.node.path);
        mac
    }
}

pub fn make_crate_like_module(mut item: Item, cx: &mut ExtCtxt) -> Item {
    // Add the module as a prefix on all absolute paths
    let mut folder = PathFolder {
        root: item.ident
    };
    item = folder.fold_item_simple(item);

    // Create a `use std` item
    let std_i = Ident::from_str("std");
    let use_std = Item {
        ident: keywords::Invalid.ident(),
        attrs: Vec::new(),
        id: cx.resolver.next_node_id(),
        node: ItemKind::Use(P(UseTree {
            span: DUMMY_SP,
            kind: UseTreeKind::Simple(Some(std_i)),
            prefix: Path {
                segments: vec![PathSegment {
                    identifier: std_i,
                    span: DUMMY_SP,
                    parameters: None,
                }],
                span: DUMMY_SP,
            },
        })),
        vis: dummy_spanned(VisibilityKind::Inherited),
        span: DUMMY_SP,
        tokens: None,
    };

    match item.node {
        ItemKind::Mod(ref mut module) => {
            // Add the `use std` item to the module
            module.items.push(P(use_std.clone()));

            // Make `main` public
            let main = Symbol::intern("main");
            for mut item in &mut module.items {
                if item.ident.name == main {
                    item.vis = dummy_spanned(VisibilityKind::Public);
                }
            }
        }
        _ => panic!("expected module"),
    }
    item
}
