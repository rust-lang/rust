// Copyright 2012-2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// FIXME (#14132): Even this SVH computation still has implementation
// artifacts: namely, the order of item declaration will affect the
// hash computation, but for many kinds of items the order of
// declaration should be irrelevant to the ABI.

pub use self::SawExprComponent::*;
pub use self::SawStmtComponent::*;
use self::SawAbiComponent::*;
use syntax::ast::{self, Name, NodeId};
use syntax::parse::token;
use syntax_pos::Span;
use rustc::hir;
use rustc::hir::*;
use rustc::hir::def::{Def, PathResolution};
use rustc::hir::def_id::DefId;
use rustc::hir::intravisit as visit;
use rustc::hir::intravisit::{Visitor, FnKind};
use rustc::ty::TyCtxt;

use std::hash::{Hash, SipHasher};

use super::def_path_hash::DefPathHashes;

pub struct StrictVersionHashVisitor<'a, 'hash: 'a, 'tcx: 'hash> {
    pub tcx: TyCtxt<'hash, 'tcx, 'tcx>,
    pub st: &'a mut SipHasher,

    // collect a deterministic hash of def-ids that we have seen
    def_path_hashes: &'a mut DefPathHashes<'hash, 'tcx>,
}

impl<'a, 'hash, 'tcx> StrictVersionHashVisitor<'a, 'hash, 'tcx> {
    pub fn new(st: &'a mut SipHasher,
               tcx: TyCtxt<'hash, 'tcx, 'tcx>,
               def_path_hashes: &'a mut DefPathHashes<'hash, 'tcx>)
               -> Self {
        StrictVersionHashVisitor { st: st, tcx: tcx, def_path_hashes: def_path_hashes }
    }

    fn compute_def_id_hash(&mut self, def_id: DefId) -> u64 {
        self.def_path_hashes.hash(def_id)
    }
}

// To off-load the bulk of the hash-computation on #[derive(Hash)],
// we define a set of enums corresponding to the content that our
// crate visitor will encounter as it traverses the ast.
//
// The important invariant is that all of the Saw*Component enums
// do not carry any Spans, Names, or Idents.
//
// Not carrying any Names/Idents is the important fix for problem
// noted on PR #13948: using the ident.name as the basis for a
// hash leads to unstable SVH, because ident.name is just an index
// into intern table (i.e. essentially a random address), not
// computed from the name content.
//
// With the below enums, the SVH computation is not sensitive to
// artifacts of how rustc was invoked nor of how the source code
// was laid out.  (Or at least it is *less* sensitive.)

// This enum represents the different potential bits of code the
// visitor could encounter that could affect the ABI for the crate,
// and assigns each a distinct tag to feed into the hash computation.
#[derive(Hash)]
enum SawAbiComponent<'a> {

    // FIXME (#14132): should we include (some function of)
    // ident.ctxt as well?
    SawIdent(token::InternedString),
    SawStructDef(token::InternedString),

    SawLifetime(token::InternedString),
    SawLifetimeDef(token::InternedString),

    SawMod,
    SawForeignItem,
    SawItem,
    SawTy,
    SawGenerics,
    SawFn,
    SawTraitItem,
    SawImplItem,
    SawStructField,
    SawVariant,
    SawPath,
    SawBlock,
    SawPat,
    SawLocal,
    SawArm,
    SawExpr(SawExprComponent<'a>),
    SawStmt(SawStmtComponent),
}

/// SawExprComponent carries all of the information that we want
/// to include in the hash that *won't* be covered by the
/// subsequent recursive traversal of the expression's
/// substructure by the visitor.
///
/// We know every Expr_ variant is covered by a variant because
/// `fn saw_expr` maps each to some case below.  Ensuring that
/// each variant carries an appropriate payload has to be verified
/// by hand.
///
/// (However, getting that *exactly* right is not so important
/// because the SVH is just a developer convenience; there is no
/// guarantee of collision-freedom, hash collisions are just
/// (hopefully) unlikely.)
#[derive(Hash)]
pub enum SawExprComponent<'a> {

    SawExprLoop(Option<token::InternedString>),
    SawExprField(token::InternedString),
    SawExprTupField(usize),
    SawExprBreak(Option<token::InternedString>),
    SawExprAgain(Option<token::InternedString>),

    SawExprBox,
    SawExprVec,
    SawExprCall,
    SawExprMethodCall,
    SawExprTup,
    SawExprBinary(hir::BinOp_),
    SawExprUnary(hir::UnOp),
    SawExprLit(ast::LitKind),
    SawExprCast,
    SawExprType,
    SawExprIf,
    SawExprWhile,
    SawExprMatch,
    SawExprClosure,
    SawExprBlock,
    SawExprAssign,
    SawExprAssignOp(hir::BinOp_),
    SawExprIndex,
    SawExprPath(Option<usize>),
    SawExprAddrOf(hir::Mutability),
    SawExprRet,
    SawExprInlineAsm(&'a hir::InlineAsm),
    SawExprStruct,
    SawExprRepeat,
}

fn saw_expr<'a>(node: &'a Expr_) -> SawExprComponent<'a> {
    match *node {
        ExprBox(..)              => SawExprBox,
        ExprVec(..)              => SawExprVec,
        ExprCall(..)             => SawExprCall,
        ExprMethodCall(..)       => SawExprMethodCall,
        ExprTup(..)              => SawExprTup,
        ExprBinary(op, ..)       => SawExprBinary(op.node),
        ExprUnary(op, _)         => SawExprUnary(op),
        ExprLit(ref lit)         => SawExprLit(lit.node.clone()),
        ExprCast(..)             => SawExprCast,
        ExprType(..)             => SawExprType,
        ExprIf(..)               => SawExprIf,
        ExprWhile(..)            => SawExprWhile,
        ExprLoop(_, id)          => SawExprLoop(id.map(|id| id.node.as_str())),
        ExprMatch(..)            => SawExprMatch,
        ExprClosure(..)          => SawExprClosure,
        ExprBlock(..)            => SawExprBlock,
        ExprAssign(..)           => SawExprAssign,
        ExprAssignOp(op, ..)     => SawExprAssignOp(op.node),
        ExprField(_, name)       => SawExprField(name.node.as_str()),
        ExprTupField(_, id)      => SawExprTupField(id.node),
        ExprIndex(..)            => SawExprIndex,
        ExprPath(ref qself, _)   => SawExprPath(qself.as_ref().map(|q| q.position)),
        ExprAddrOf(m, _)         => SawExprAddrOf(m),
        ExprBreak(id)            => SawExprBreak(id.map(|id| id.node.as_str())),
        ExprAgain(id)            => SawExprAgain(id.map(|id| id.node.as_str())),
        ExprRet(..)              => SawExprRet,
        ExprInlineAsm(ref a,..)  => SawExprInlineAsm(a),
        ExprStruct(..)           => SawExprStruct,
        ExprRepeat(..)           => SawExprRepeat,
    }
}

/// SawStmtComponent is analogous to SawExprComponent, but for statements.
#[derive(Hash)]
pub enum SawStmtComponent {
    SawStmtExpr,
    SawStmtSemi,
}

impl<'a, 'hash, 'tcx> Visitor<'tcx> for StrictVersionHashVisitor<'a, 'hash, 'tcx> {
    fn visit_nested_item(&mut self, _: ItemId) {
        // Each item is hashed independently; ignore nested items.
    }

    fn visit_variant_data(&mut self, s: &'tcx VariantData, name: Name,
                          g: &'tcx Generics, _: NodeId, _: Span) {
        debug!("visit_variant_data: st={:?}", self.st);
        SawStructDef(name.as_str()).hash(self.st);
        visit::walk_generics(self, g);
        visit::walk_struct_def(self, s)
    }

    fn visit_variant(&mut self, v: &'tcx Variant, g: &'tcx Generics, item_id: NodeId) {
        debug!("visit_variant: st={:?}", self.st);
        SawVariant.hash(self.st);
        // walk_variant does not call walk_generics, so do it here.
        visit::walk_generics(self, g);
        visit::walk_variant(self, v, g, item_id)
    }

    // All of the remaining methods just record (in the hash
    // SipHasher) that the visitor saw that particular variant
    // (with its payload), and continue walking as the default
    // visitor would.
    //
    // Some of the implementations have some notes as to how one
    // might try to make their SVH computation less discerning
    // (e.g. by incorporating reachability analysis).  But
    // currently all of their implementations are uniform and
    // uninteresting.
    //
    // (If you edit a method such that it deviates from the
    // pattern, please move that method up above this comment.)

    fn visit_name(&mut self, _: Span, name: Name) {
        debug!("visit_name: st={:?}", self.st);
        SawIdent(name.as_str()).hash(self.st);
    }

    fn visit_lifetime(&mut self, l: &'tcx Lifetime) {
        debug!("visit_lifetime: st={:?}", self.st);
        SawLifetime(l.name.as_str()).hash(self.st);
    }

    fn visit_lifetime_def(&mut self, l: &'tcx LifetimeDef) {
        debug!("visit_lifetime_def: st={:?}", self.st);
        SawLifetimeDef(l.lifetime.name.as_str()).hash(self.st);
    }

    // We do recursively walk the bodies of functions/methods
    // (rather than omitting their bodies from the hash) since
    // monomorphization and cross-crate inlining generally implies
    // that a change to a crate body will require downstream
    // crates to be recompiled.
    fn visit_expr(&mut self, ex: &'tcx Expr) {
        debug!("visit_expr: st={:?}", self.st);
        SawExpr(saw_expr(&ex.node)).hash(self.st); visit::walk_expr(self, ex)
    }

    fn visit_stmt(&mut self, s: &'tcx Stmt) {
        debug!("visit_stmt: st={:?}", self.st);

        // We don't want to modify the hash for decls, because
        // they might be item decls (if they are local decls,
        // we'll hash that fact in visit_local); but we do want to
        // remember if this was a StmtExpr or StmtSemi (the later
        // had an explicit semi-colon; this affects the typing
        // rules).
        match s.node {
            StmtDecl(..) => (),
            StmtExpr(..) => SawStmt(SawStmtExpr).hash(self.st),
            StmtSemi(..) => SawStmt(SawStmtSemi).hash(self.st),
        }

        visit::walk_stmt(self, s)
    }

    fn visit_foreign_item(&mut self, i: &'tcx ForeignItem) {
        debug!("visit_foreign_item: st={:?}", self.st);

        // FIXME (#14132) ideally we would incorporate privacy (or
        // perhaps reachability) somewhere here, so foreign items
        // that do not leak into downstream crates would not be
        // part of the ABI.
        SawForeignItem.hash(self.st); visit::walk_foreign_item(self, i)
    }

    fn visit_item(&mut self, i: &'tcx Item) {
        debug!("visit_item: {:?} st={:?}", i, self.st);

        // FIXME (#14132) ideally would incorporate reachability
        // analysis somewhere here, so items that never leak into
        // downstream crates (e.g. via monomorphisation or
        // inlining) would not be part of the ABI.
        SawItem.hash(self.st); visit::walk_item(self, i)
    }

    fn visit_mod(&mut self, m: &'tcx Mod, _s: Span, n: NodeId) {
        debug!("visit_mod: st={:?}", self.st);
        SawMod.hash(self.st); visit::walk_mod(self, m, n)
    }

    fn visit_ty(&mut self, t: &'tcx Ty) {
        debug!("visit_ty: st={:?}", self.st);
        SawTy.hash(self.st); visit::walk_ty(self, t)
    }

    fn visit_generics(&mut self, g: &'tcx Generics) {
        debug!("visit_generics: st={:?}", self.st);
        SawGenerics.hash(self.st); visit::walk_generics(self, g)
    }

    fn visit_fn(&mut self, fk: FnKind<'tcx>, fd: &'tcx FnDecl,
                b: &'tcx Block, s: Span, n: NodeId) {
        debug!("visit_fn: st={:?}", self.st);
        SawFn.hash(self.st); visit::walk_fn(self, fk, fd, b, s, n)
    }

    fn visit_trait_item(&mut self, ti: &'tcx TraitItem) {
        debug!("visit_trait_item: st={:?}", self.st);
        SawTraitItem.hash(self.st); visit::walk_trait_item(self, ti)
    }

    fn visit_impl_item(&mut self, ii: &'tcx ImplItem) {
        debug!("visit_impl_item: st={:?}", self.st);
        SawImplItem.hash(self.st); visit::walk_impl_item(self, ii)
    }

    fn visit_struct_field(&mut self, s: &'tcx StructField) {
        debug!("visit_struct_field: st={:?}", self.st);
        SawStructField.hash(self.st); visit::walk_struct_field(self, s)
    }

    fn visit_path(&mut self, path: &'tcx Path, _: ast::NodeId) {
        debug!("visit_path: st={:?}", self.st);
        SawPath.hash(self.st); visit::walk_path(self, path)
    }

    fn visit_block(&mut self, b: &'tcx Block) {
        debug!("visit_block: st={:?}", self.st);
        SawBlock.hash(self.st); visit::walk_block(self, b)
    }

    fn visit_pat(&mut self, p: &'tcx Pat) {
        debug!("visit_pat: st={:?}", self.st);
        SawPat.hash(self.st); visit::walk_pat(self, p)
    }

    fn visit_local(&mut self, l: &'tcx Local) {
        debug!("visit_local: st={:?}", self.st);
        SawLocal.hash(self.st); visit::walk_local(self, l)
    }

    fn visit_arm(&mut self, a: &'tcx Arm) {
        debug!("visit_arm: st={:?}", self.st);
        SawArm.hash(self.st); visit::walk_arm(self, a)
    }

    fn visit_id(&mut self, id: NodeId) {
        debug!("visit_id: id={} st={:?}", id, self.st);
        self.hash_resolve(id);
    }
}

#[derive(Hash)]
pub enum DefHash {
    SawDefId,
    SawLabel,
    SawPrimTy,
    SawSelfTy,
    SawErr,
}

impl<'a, 'hash, 'tcx> StrictVersionHashVisitor<'a, 'hash, 'tcx> {
    fn hash_resolve(&mut self, id: ast::NodeId) {
        // Because whether or not a given id has an entry is dependent
        // solely on expr variant etc, we don't need to hash whether
        // or not an entry was present (we are already hashing what
        // variant it is above when we visit the HIR).

        if let Some(def) = self.tcx.def_map.borrow().get(&id) {
            debug!("hash_resolve: id={:?} def={:?} st={:?}", id, def, self.st);
            self.hash_partial_def(def);
        }

        if let Some(traits) = self.tcx.trait_map.get(&id) {
            debug!("hash_resolve: id={:?} traits={:?} st={:?}", id, traits, self.st);
            traits.len().hash(self.st);

            // The ordering of the candidates is not fixed. So we hash
            // the def-ids and then sort them and hash the collection.
            let mut candidates: Vec<_> =
                traits.iter()
                      .map(|&TraitCandidate { def_id, import_id: _ }| {
                          self.compute_def_id_hash(def_id)
                      })
                      .collect();
            candidates.sort();
            candidates.hash(self.st);
        }
    }

    fn hash_def_id(&mut self, def_id: DefId) {
        self.compute_def_id_hash(def_id).hash(self.st);
    }

    fn hash_partial_def(&mut self, def: &PathResolution) {
        self.hash_def(def.base_def);
        def.depth.hash(self.st);
    }

    fn hash_def(&mut self, def: Def) {
        match def {
            // Crucial point: for all of these variants, the variant +
            // add'l data that is added is always the same if the
            // def-id is the same, so it suffices to hash the def-id
            Def::Fn(..) |
            Def::Mod(..) |
            Def::ForeignMod(..) |
            Def::Static(..) |
            Def::Variant(..) |
            Def::Enum(..) |
            Def::TyAlias(..) |
            Def::AssociatedTy(..) |
            Def::TyParam(..) |
            Def::Struct(..) |
            Def::Union(..) |
            Def::Trait(..) |
            Def::Method(..) |
            Def::Const(..) |
            Def::AssociatedConst(..) |
            Def::Local(..) |
            Def::Upvar(..) => {
                DefHash::SawDefId.hash(self.st);
                self.hash_def_id(def.def_id());
            }

            Def::Label(..) => {
                DefHash::SawLabel.hash(self.st);
                // we don't encode the `id` because it always refers to something
                // within this item, so if it changed, there would have to be other
                // changes too
            }
            Def::PrimTy(ref prim_ty) => {
                DefHash::SawPrimTy.hash(self.st);
                prim_ty.hash(self.st);
            }
            Def::SelfTy(..) => {
                DefHash::SawSelfTy.hash(self.st);
                // the meaning of Self is always the same within a
                // given context, so we don't need to hash the other
                // fields
            }
            Def::Err => {
                DefHash::SawErr.hash(self.st);
            }
        }
    }
}
