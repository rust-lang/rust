// Copyright 2012-2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! Calculation and management of a Strict Version Hash for crates
//!
//! # Today's ABI problem
//!
//! In today's implementation of rustc, it is incredibly difficult to achieve
//! forward binary compatibility without resorting to C-like interfaces. Within
//! rust code itself, abi details such as symbol names suffer from a variety of
//! unrelated factors to code changing such as the "def id drift" problem. This
//! ends up yielding confusing error messages about metadata mismatches and
//! such.
//!
//! The core of this problem is when an upstream dependency changes and
//! downstream dependents are not recompiled. This causes compile errors because
//! the upstream crate's metadata has changed but the downstream crates are
//! still referencing the older crate's metadata.
//!
//! This problem exists for many reasons, the primary of which is that rust does
//! not currently support forwards ABI compatibility (in place upgrades of a
//! crate).
//!
//! # SVH and how it alleviates the problem
//!
//! With all of this knowledge on hand, this module contains the implementation
//! of a notion of a "Strict Version Hash" for a crate. This is essentially a
//! hash of all contents of a crate which can somehow be exposed to downstream
//! crates.
//!
//! This hash is currently calculated by just hashing the AST, but this is
//! obviously wrong (doc changes should not result in an incompatible ABI).
//! Implementation-wise, this is required at this moment in time.
//!
//! By encoding this strict version hash into all crate's metadata, stale crates
//! can be detected immediately and error'd about by rustc itself.
//!
//! # Relevant links
//!
//! Original issue: https://github.com/rust-lang/rust/issues/10207

use std::fmt;
use std::hash::{Hash, SipHasher, Hasher};
use rustc_front::hir;
use rustc_front::intravisit as visit;

#[derive(Clone, PartialEq, Debug)]
pub struct Svh {
    hash: String,
}

impl Svh {
    pub fn new(hash: &str) -> Svh {
        assert!(hash.len() == 16);
        Svh { hash: hash.to_string() }
    }

    pub fn as_str<'a>(&'a self) -> &'a str {
        &self.hash
    }

    pub fn calculate(metadata: &Vec<String>, krate: &hir::Crate) -> Svh {
        fn hex(b: u64) -> char {
            let b = (b & 0xf) as u8;
            let b = match b {
                0 ... 9 => '0' as u8 + b,
                _ => 'a' as u8 + b - 10,
            };
            b as char
        }

        // FIXME (#14132): This is better than it used to be, but it still not
        // ideal. We now attempt to hash only the relevant portions of the
        // Crate AST as well as the top-level crate attributes. (However,
        // the hashing of the crate attributes should be double-checked
        // to ensure it is not incorporating implementation artifacts into
        // the hash that are not otherwise visible.)

        // FIXME: this should use SHA1, not SipHash. SipHash is not built to
        //        avoid collisions.
        let mut state = SipHasher::new();

        for data in metadata {
            data.hash(&mut state);
        }

        {
            let mut visit = svh_visitor::make(&mut state, krate);
            visit::walk_crate(&mut visit, krate);
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
            attr.node.value.hash(&mut state);
        }

        let hash = state.finish();
        Svh {
            hash: (0..64).step_by(4).map(|i| hex(hash >> i)).collect()
        }
    }
}

impl fmt::Display for Svh {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        f.pad(self.as_str())
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
    use syntax::codemap::Span;
    use syntax::parse::token;
    use rustc_front::intravisit as visit;
    use rustc_front::intravisit::{Visitor, FnKind};
    use rustc_front::hir::*;
    use rustc_front::hir;

    use std::hash::{Hash, SipHasher};

    pub struct StrictVersionHashVisitor<'a> {
        pub krate: &'a Crate,
        pub st: &'a mut SipHasher,
    }

    pub fn make<'a>(st: &'a mut SipHasher, krate: &'a Crate) -> StrictVersionHashVisitor<'a> {
        StrictVersionHashVisitor { st: st, krate: krate }
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
        SawExplicitSelf,
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
        SawExprLit(ast::Lit_),
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
        SawExprRange,
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
            ExprLoop(_, id)          => SawExprLoop(id.map(|id| id.name.as_str())),
            ExprMatch(..)            => SawExprMatch,
            ExprClosure(..)          => SawExprClosure,
            ExprBlock(..)            => SawExprBlock,
            ExprAssign(..)           => SawExprAssign,
            ExprAssignOp(op, _, _)   => SawExprAssignOp(op.node),
            ExprField(_, name)       => SawExprField(name.node.as_str()),
            ExprTupField(_, id)      => SawExprTupField(id.node),
            ExprIndex(..)            => SawExprIndex,
            ExprRange(..)            => SawExprRange,
            ExprPath(ref qself, _)   => SawExprPath(qself.as_ref().map(|q| q.position)),
            ExprAddrOf(m, _)         => SawExprAddrOf(m),
            ExprBreak(id)            => SawExprBreak(id.map(|id| id.node.name.as_str())),
            ExprAgain(id)            => SawExprAgain(id.map(|id| id.node.name.as_str())),
            ExprRet(..)              => SawExprRet,
            ExprInlineAsm(ref asm)   => SawExprInlineAsm(asm),
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

    impl<'a> Visitor<'a> for StrictVersionHashVisitor<'a> {
        fn visit_nested_item(&mut self, item: ItemId) {
            self.visit_item(self.krate.item(item.id))
        }

        fn visit_variant_data(&mut self, s: &'a VariantData, name: Name,
                              g: &'a Generics, _: NodeId, _: Span) {
            SawStructDef(name.as_str()).hash(self.st);
            visit::walk_generics(self, g);
            visit::walk_struct_def(self, s)
        }

        fn visit_variant(&mut self, v: &'a Variant, g: &'a Generics, item_id: NodeId) {
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
            SawIdent(name.as_str()).hash(self.st);
        }

        fn visit_lifetime(&mut self, l: &'a Lifetime) {
            SawLifetime(l.name.as_str()).hash(self.st);
        }

        fn visit_lifetime_def(&mut self, l: &'a LifetimeDef) {
            SawLifetimeDef(l.lifetime.name.as_str()).hash(self.st);
        }

        // We do recursively walk the bodies of functions/methods
        // (rather than omitting their bodies from the hash) since
        // monomorphization and cross-crate inlining generally implies
        // that a change to a crate body will require downstream
        // crates to be recompiled.
        fn visit_expr(&mut self, ex: &'a Expr) {
            SawExpr(saw_expr(&ex.node)).hash(self.st); visit::walk_expr(self, ex)
        }

        fn visit_stmt(&mut self, s: &'a Stmt) {
            SawStmt(saw_stmt(&s.node)).hash(self.st); visit::walk_stmt(self, s)
        }

        fn visit_foreign_item(&mut self, i: &'a ForeignItem) {
            // FIXME (#14132) ideally we would incorporate privacy (or
            // perhaps reachability) somewhere here, so foreign items
            // that do not leak into downstream crates would not be
            // part of the ABI.
            SawForeignItem.hash(self.st); visit::walk_foreign_item(self, i)
        }

        fn visit_item(&mut self, i: &'a Item) {
            // FIXME (#14132) ideally would incorporate reachability
            // analysis somewhere here, so items that never leak into
            // downstream crates (e.g. via monomorphisation or
            // inlining) would not be part of the ABI.
            SawItem.hash(self.st); visit::walk_item(self, i)
        }

        fn visit_mod(&mut self, m: &'a Mod, _s: Span, _n: NodeId) {
            SawMod.hash(self.st); visit::walk_mod(self, m)
        }

        fn visit_decl(&mut self, d: &'a Decl) {
            SawDecl.hash(self.st); visit::walk_decl(self, d)
        }

        fn visit_ty(&mut self, t: &'a Ty) {
            SawTy.hash(self.st); visit::walk_ty(self, t)
        }

        fn visit_generics(&mut self, g: &'a Generics) {
            SawGenerics.hash(self.st); visit::walk_generics(self, g)
        }

        fn visit_fn(&mut self, fk: FnKind<'a>, fd: &'a FnDecl,
                    b: &'a Block, s: Span, _: NodeId) {
            SawFn.hash(self.st); visit::walk_fn(self, fk, fd, b, s)
        }

        fn visit_trait_item(&mut self, ti: &'a TraitItem) {
            SawTraitItem.hash(self.st); visit::walk_trait_item(self, ti)
        }

        fn visit_impl_item(&mut self, ii: &'a ImplItem) {
            SawImplItem.hash(self.st); visit::walk_impl_item(self, ii)
        }

        fn visit_struct_field(&mut self, s: &'a StructField) {
            SawStructField.hash(self.st); visit::walk_struct_field(self, s)
        }

        fn visit_explicit_self(&mut self, es: &'a ExplicitSelf) {
            SawExplicitSelf.hash(self.st); visit::walk_explicit_self(self, es)
        }

        fn visit_path(&mut self, path: &'a Path, _: ast::NodeId) {
            SawPath.hash(self.st); visit::walk_path(self, path)
        }

        fn visit_path_list_item(&mut self, prefix: &'a Path, item: &'a PathListItem) {
            SawPath.hash(self.st); visit::walk_path_list_item(self, prefix, item)
        }

        fn visit_block(&mut self, b: &'a Block) {
            SawBlock.hash(self.st); visit::walk_block(self, b)
        }

        fn visit_pat(&mut self, p: &'a Pat) {
            SawPat.hash(self.st); visit::walk_pat(self, p)
        }

        fn visit_local(&mut self, l: &'a Local) {
            SawLocal.hash(self.st); visit::walk_local(self, l)
        }

        fn visit_arm(&mut self, a: &'a Arm) {
            SawArm.hash(self.st); visit::walk_arm(self, a)
        }
    }
}
