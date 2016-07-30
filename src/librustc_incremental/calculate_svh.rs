// Copyright 2012-2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! Calculation of a Strict Version Hash for crates.  For a length
//! comment explaining the general idea, see `librustc/middle/svh.rs`.

use syntax::attr::AttributeMethods;
use std::hash::{Hash, SipHasher, Hasher};
use rustc::hir::def_id::{CRATE_DEF_INDEX, DefId};
use rustc::hir::svh::Svh;
use rustc::ty::TyCtxt;
use rustc::hir::intravisit::{self, Visitor};

use self::svh_visitor::StrictVersionHashVisitor;

pub trait SvhCalculate {
    /// Calculate the SVH for an entire krate.
    fn calculate_krate_hash(self) -> Svh;

    /// Calculate the SVH for a particular item.
    fn calculate_item_hash(self, def_id: DefId) -> u64;
}

impl<'a, 'tcx> SvhCalculate for TyCtxt<'a, 'tcx, 'tcx> {
    fn calculate_krate_hash(self) -> Svh {
        // FIXME (#14132): This is better than it used to be, but it still not
        // ideal. We now attempt to hash only the relevant portions of the
        // Crate AST as well as the top-level crate attributes. (However,
        // the hashing of the crate attributes should be double-checked
        // to ensure it is not incorporating implementation artifacts into
        // the hash that are not otherwise visible.)

        let crate_disambiguator = self.sess.local_crate_disambiguator();
        let krate = self.map.krate();

        // FIXME: this should use SHA1, not SipHash. SipHash is not built to
        //        avoid collisions.
        let mut state = SipHasher::new();
        debug!("state: {:?}", state);

        // FIXME(#32753) -- at (*) we `to_le` for endianness, but is
        // this enough, and does it matter anyway?
        "crate_disambiguator".hash(&mut state);
        crate_disambiguator.len().to_le().hash(&mut state); // (*)
        crate_disambiguator.hash(&mut state);

        debug!("crate_disambiguator: {:?}", crate_disambiguator);
        debug!("state: {:?}", state);

        {
            let mut visit = StrictVersionHashVisitor::new(&mut state, self);
            krate.visit_all_items(&mut visit);
        }

        // FIXME (#14132): This hash is still sensitive to e.g. the
        // spans of the crate Attributes and their underlying
        // MetaItems; we should make ContentHashable impl for those
        // types and then use hash_content.  But, since all crate
        // attributes should appear near beginning of the file, it is
        // not such a big deal to be sensitive to their spans for now.
        //
        // We hash only the MetaItems instead of the entire Attribute
        // to avoid hashing the AttrId
        for attr in &krate.attrs {
            debug!("krate attr {:?}", attr);
            attr.meta().hash(&mut state);
        }

        Svh::new(state.finish())
    }

    fn calculate_item_hash(self, def_id: DefId) -> u64 {
        assert!(def_id.is_local());

        debug!("calculate_item_hash(def_id={:?})", def_id);

        let mut state = SipHasher::new();

        {
            let mut visit = StrictVersionHashVisitor::new(&mut state, self);
            if def_id.index == CRATE_DEF_INDEX {
                // the crate root itself is not registered in the map
                // as an item, so we have to fetch it this way
                let krate = self.map.krate();
                intravisit::walk_crate(&mut visit, krate);
            } else {
                let node_id = self.map.as_local_node_id(def_id).unwrap();
                let item = self.map.expect_item(node_id);
                visit.visit_item(item);
            }
        }

        let hash = state.finish();

        debug!("calculate_item_hash: def_id={:?} hash={:?}", def_id, hash);

        hash
    }
}

// FIXME (#14132): Even this SVH computation still has implementation
// artifacts: namely, the order of item declaration will affect the
// hash computation, but for many kinds of items the order of
// declaration should be irrelevant to the ABI.

mod svh_visitor {
    pub use self::SawExprComponent::*;
    pub use self::SawStmtComponent::*;
    use self::SawAbiComponent::*;
    use syntax::ast::{self, Name, NodeId};
    use syntax::parse::token;
    use syntax_pos::Span;
    use rustc::ty::TyCtxt;
    use rustc::hir;
    use rustc::hir::*;
    use rustc::hir::map::DefPath;
    use rustc::hir::intravisit as visit;
    use rustc::hir::intravisit::{Visitor, FnKind};

    use std::hash::{Hash, SipHasher};

    pub struct StrictVersionHashVisitor<'a, 'tcx: 'a> {
        pub tcx: TyCtxt<'a, 'tcx, 'tcx>,
        pub st: &'a mut SipHasher,
    }

    impl<'a, 'tcx> StrictVersionHashVisitor<'a, 'tcx> {
        pub fn new(st: &'a mut SipHasher,
                   tcx: TyCtxt<'a, 'tcx, 'tcx>)
                   -> Self {
            StrictVersionHashVisitor { st: st, tcx: tcx }
        }

        fn hash_def_path(&mut self, path: &DefPath) {
            self.tcx.crate_name(path.krate).hash(self.st);
            self.tcx.crate_disambiguator(path.krate).hash(self.st);
            for data in &path.data {
                data.data.as_interned_str().hash(self.st);
                data.disambiguator.hash(self.st);
            }
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
        SawDecl,
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
            ExprBinary(op, _, _)     => SawExprBinary(op.node),
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
            ExprAssignOp(op, _, _)   => SawExprAssignOp(op.node),
            ExprField(_, name)       => SawExprField(name.node.as_str()),
            ExprTupField(_, id)      => SawExprTupField(id.node),
            ExprIndex(..)            => SawExprIndex,
            ExprPath(ref qself, _)   => SawExprPath(qself.as_ref().map(|q| q.position)),
            ExprAddrOf(m, _)         => SawExprAddrOf(m),
            ExprBreak(id)            => SawExprBreak(id.map(|id| id.node.as_str())),
            ExprAgain(id)            => SawExprAgain(id.map(|id| id.node.as_str())),
            ExprRet(..)              => SawExprRet,
            ExprInlineAsm(ref a,_,_) => SawExprInlineAsm(a),
            ExprStruct(..)           => SawExprStruct,
            ExprRepeat(..)           => SawExprRepeat,
        }
    }

    /// SawStmtComponent is analogous to SawExprComponent, but for statements.
    #[derive(Hash)]
    pub enum SawStmtComponent {
        SawStmtDecl,
        SawStmtExpr,
        SawStmtSemi,
    }

    fn saw_stmt(node: &Stmt_) -> SawStmtComponent {
        match *node {
            StmtDecl(..) => SawStmtDecl,
            StmtExpr(..) => SawStmtExpr,
            StmtSemi(..) => SawStmtSemi,
        }
    }

    impl<'a, 'tcx> Visitor<'a> for StrictVersionHashVisitor<'a, 'tcx> {
        fn visit_nested_item(&mut self, item: ItemId) {
            let def_path = self.tcx.map.def_path_from_id(item.id).unwrap();
            debug!("visit_nested_item: def_path={:?} st={:?}", def_path, self.st);
            self.hash_def_path(&def_path);
        }

        fn visit_variant_data(&mut self, s: &'a VariantData, name: Name,
                              g: &'a Generics, _: NodeId, _: Span) {
            debug!("visit_variant_data: st={:?}", self.st);
            SawStructDef(name.as_str()).hash(self.st);
            visit::walk_generics(self, g);
            visit::walk_struct_def(self, s)
        }

        fn visit_variant(&mut self, v: &'a Variant, g: &'a Generics, item_id: NodeId) {
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

        fn visit_lifetime(&mut self, l: &'a Lifetime) {
            debug!("visit_lifetime: st={:?}", self.st);
            SawLifetime(l.name.as_str()).hash(self.st);
        }

        fn visit_lifetime_def(&mut self, l: &'a LifetimeDef) {
            debug!("visit_lifetime_def: st={:?}", self.st);
            SawLifetimeDef(l.lifetime.name.as_str()).hash(self.st);
        }

        // We do recursively walk the bodies of functions/methods
        // (rather than omitting their bodies from the hash) since
        // monomorphization and cross-crate inlining generally implies
        // that a change to a crate body will require downstream
        // crates to be recompiled.
        fn visit_expr(&mut self, ex: &'a Expr) {
            debug!("visit_expr: st={:?}", self.st);
            SawExpr(saw_expr(&ex.node)).hash(self.st); visit::walk_expr(self, ex)
        }

        fn visit_stmt(&mut self, s: &'a Stmt) {
            debug!("visit_stmt: st={:?}", self.st);
            SawStmt(saw_stmt(&s.node)).hash(self.st); visit::walk_stmt(self, s)
        }

        fn visit_foreign_item(&mut self, i: &'a ForeignItem) {
            debug!("visit_foreign_item: st={:?}", self.st);

            // FIXME (#14132) ideally we would incorporate privacy (or
            // perhaps reachability) somewhere here, so foreign items
            // that do not leak into downstream crates would not be
            // part of the ABI.
            SawForeignItem.hash(self.st); visit::walk_foreign_item(self, i)
        }

        fn visit_item(&mut self, i: &'a Item) {
            debug!("visit_item: {:?} st={:?}", i, self.st);

            // FIXME (#14132) ideally would incorporate reachability
            // analysis somewhere here, so items that never leak into
            // downstream crates (e.g. via monomorphisation or
            // inlining) would not be part of the ABI.
            SawItem.hash(self.st); visit::walk_item(self, i)
        }

        fn visit_mod(&mut self, m: &'a Mod, _s: Span, n: NodeId) {
            debug!("visit_mod: st={:?}", self.st);
            SawMod.hash(self.st); visit::walk_mod(self, m, n)
        }

        fn visit_decl(&mut self, d: &'a Decl) {
            debug!("visit_decl: st={:?}", self.st);
            SawDecl.hash(self.st); visit::walk_decl(self, d)
        }

        fn visit_ty(&mut self, t: &'a Ty) {
            debug!("visit_ty: st={:?}", self.st);
            SawTy.hash(self.st); visit::walk_ty(self, t)
        }

        fn visit_generics(&mut self, g: &'a Generics) {
            debug!("visit_generics: st={:?}", self.st);
            SawGenerics.hash(self.st); visit::walk_generics(self, g)
        }

        fn visit_fn(&mut self, fk: FnKind<'a>, fd: &'a FnDecl,
                    b: &'a Block, s: Span, n: NodeId) {
            debug!("visit_fn: st={:?}", self.st);
            SawFn.hash(self.st); visit::walk_fn(self, fk, fd, b, s, n)
        }

        fn visit_trait_item(&mut self, ti: &'a TraitItem) {
            debug!("visit_trait_item: st={:?}", self.st);
            SawTraitItem.hash(self.st); visit::walk_trait_item(self, ti)
        }

        fn visit_impl_item(&mut self, ii: &'a ImplItem) {
            debug!("visit_impl_item: st={:?}", self.st);
            SawImplItem.hash(self.st); visit::walk_impl_item(self, ii)
        }

        fn visit_struct_field(&mut self, s: &'a StructField) {
            debug!("visit_struct_field: st={:?}", self.st);
            SawStructField.hash(self.st); visit::walk_struct_field(self, s)
        }

        fn visit_path(&mut self, path: &'a Path, _: ast::NodeId) {
            debug!("visit_path: st={:?}", self.st);
            SawPath.hash(self.st); visit::walk_path(self, path)
        }

        fn visit_block(&mut self, b: &'a Block) {
            debug!("visit_block: st={:?}", self.st);
            SawBlock.hash(self.st); visit::walk_block(self, b)
        }

        fn visit_pat(&mut self, p: &'a Pat) {
            debug!("visit_pat: st={:?}", self.st);
            SawPat.hash(self.st); visit::walk_pat(self, p)
        }

        fn visit_local(&mut self, l: &'a Local) {
            debug!("visit_local: st={:?}", self.st);
            SawLocal.hash(self.st); visit::walk_local(self, l)
        }

        fn visit_arm(&mut self, a: &'a Arm) {
            debug!("visit_arm: st={:?}", self.st);
            SawArm.hash(self.st); visit::walk_arm(self, a)
        }
    }
}
