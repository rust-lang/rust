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
use std::hash::Hash;
use std::hash::sip::SipState;
use std::iter::range_step;
use syntax::ast;
use syntax::visit;

#[deriving(Clone, PartialEq)]
pub struct Svh {
    hash: String,
}

impl Svh {
    pub fn new(hash: &str) -> Svh {
        assert!(hash.len() == 16);
        Svh { hash: hash.to_string() }
    }

    pub fn as_str<'a>(&'a self) -> &'a str {
        self.hash.as_slice()
    }

    pub fn calculate(metadata: &Vec<String>, krate: &ast::Crate) -> Svh {
        // FIXME (#14132): This is better than it used to be, but it still not
        // ideal. We now attempt to hash only the relevant portions of the
        // Crate AST as well as the top-level crate attributes. (However,
        // the hashing of the crate attributes should be double-checked
        // to ensure it is not incorporating implementation artifacts into
        // the hash that are not otherwise visible.)

        // FIXME: this should use SHA1, not SipHash. SipHash is not built to
        //        avoid collisions.
        let mut state = SipState::new();

        for data in metadata.iter() {
            data.hash(&mut state);
        }

        {
            let mut visit = svh_visitor::make(&mut state);
            visit::walk_crate(&mut visit, krate, ());
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
        for attr in krate.attrs.iter() {
            attr.node.value.hash(&mut state);
        }

        let hash = state.result();
        return Svh {
            hash: range_step(0u, 64u, 4u).map(|i| hex(hash >> i)).collect()
        };

        fn hex(b: u64) -> char {
            let b = (b & 0xf) as u8;
            let b = match b {
                0 .. 9 => '0' as u8 + b,
                _ => 'a' as u8 + b - 10,
            };
            b as char
        }
    }
}

impl fmt::Show for Svh {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        f.pad(self.as_str())
    }
}

// FIXME (#14132): Even this SVH computation still has implementation
// artifacts: namely, the order of item declaration will affect the
// hash computation, but for many kinds of items the order of
// declaration should be irrelevant to the ABI.

mod svh_visitor {
    use syntax::ast;
    use syntax::ast::*;
    use syntax::codemap::Span;
    use syntax::parse::token;
    use syntax::print::pprust;
    use syntax::visit;
    use syntax::visit::{Visitor, FnKind};

    use std::hash::Hash;
    use std::hash::sip::SipState;

    pub struct StrictVersionHashVisitor<'a> {
        pub st: &'a mut SipState,
    }

    pub fn make<'a>(st: &'a mut SipState) -> StrictVersionHashVisitor<'a> {
        StrictVersionHashVisitor { st: st }
    }

    // To off-load the bulk of the hash-computation on deriving(Hash),
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
    #[deriving(Hash)]
    enum SawAbiComponent<'a> {

        // FIXME (#14132): should we include (some function of)
        // ident.ctxt as well?
        SawIdent(token::InternedString),
        SawStructDef(token::InternedString),

        SawLifetimeRef(token::InternedString),
        SawLifetimeDecl(token::InternedString),

        SawMod,
        SawViewItem,
        SawForeignItem,
        SawItem,
        SawDecl,
        SawTy,
        SawGenerics,
        SawFn,
        SawTyMethod,
        SawTraitMethod,
        SawStructField,
        SawVariant,
        SawExplicitSelf,
        SawPath,
        SawOptLifetimeRef,
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
    #[deriving(Hash)]
    pub enum SawExprComponent<'a> {

        SawExprLoop(Option<token::InternedString>),
        SawExprField(token::InternedString),
        SawExprTupField(uint),
        SawExprBreak(Option<token::InternedString>),
        SawExprAgain(Option<token::InternedString>),

        SawExprBox,
        SawExprVec,
        SawExprCall,
        SawExprMethodCall,
        SawExprTup,
        SawExprBinary(ast::BinOp),
        SawExprUnary(ast::UnOp),
        SawExprLit(ast::Lit_),
        SawExprCast,
        SawExprIf,
        SawExprWhile,
        SawExprMatch,
        SawExprFnBlock,
        SawExprUnboxedFn,
        SawExprProc,
        SawExprBlock,
        SawExprAssign,
        SawExprAssignOp(ast::BinOp),
        SawExprIndex,
        SawExprPath,
        SawExprAddrOf(ast::Mutability),
        SawExprRet,
        SawExprInlineAsm(&'a ast::InlineAsm),
        SawExprStruct,
        SawExprRepeat,
        SawExprParen,
        SawExprForLoop,
    }

    fn saw_expr<'a>(node: &'a Expr_) -> SawExprComponent<'a> {
        match *node {
            ExprBox(..)              => SawExprBox,
            ExprVec(..)              => SawExprVec,
            ExprCall(..)             => SawExprCall,
            ExprMethodCall(..)       => SawExprMethodCall,
            ExprTup(..)              => SawExprTup,
            ExprBinary(op, _, _)     => SawExprBinary(op),
            ExprUnary(op, _)         => SawExprUnary(op),
            ExprLit(lit)             => SawExprLit(lit.node.clone()),
            ExprCast(..)             => SawExprCast,
            ExprIf(..)               => SawExprIf,
            ExprWhile(..)            => SawExprWhile,
            ExprLoop(_, id)          => SawExprLoop(id.map(content)),
            ExprMatch(..)            => SawExprMatch,
            ExprFnBlock(..)          => SawExprFnBlock,
            ExprUnboxedFn(..)        => SawExprUnboxedFn,
            ExprProc(..)             => SawExprProc,
            ExprBlock(..)            => SawExprBlock,
            ExprAssign(..)           => SawExprAssign,
            ExprAssignOp(op, _, _)   => SawExprAssignOp(op),
            ExprField(_, id, _)      => SawExprField(content(id.node)),
            ExprTupField(_, id, _)   => SawExprTupField(id.node),
            ExprIndex(..)            => SawExprIndex,
            ExprPath(..)             => SawExprPath,
            ExprAddrOf(m, _)         => SawExprAddrOf(m),
            ExprBreak(id)            => SawExprBreak(id.map(content)),
            ExprAgain(id)            => SawExprAgain(id.map(content)),
            ExprRet(..)              => SawExprRet,
            ExprInlineAsm(ref asm)   => SawExprInlineAsm(asm),
            ExprStruct(..)           => SawExprStruct,
            ExprRepeat(..)           => SawExprRepeat,
            ExprParen(..)            => SawExprParen,
            ExprForLoop(..)          => SawExprForLoop,

            // just syntactic artifacts, expanded away by time of SVH.
            ExprMac(..)              => unreachable!(),
        }
    }

    /// SawStmtComponent is analogous to SawExprComponent, but for statements.
    #[deriving(Hash)]
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
            StmtMac(..)  => unreachable!(),
        }
    }

    // Ad-hoc overloading between Ident and Name to their intern table lookups.
    trait InternKey { fn get_content(self) -> token::InternedString; }
    impl InternKey for Ident {
        fn get_content(self) -> token::InternedString { token::get_ident(self) }
    }
    impl InternKey for Name {
        fn get_content(self) -> token::InternedString { token::get_name(self) }
    }
    fn content<K:InternKey>(k: K) -> token::InternedString { k.get_content() }

    // local short-hand eases writing signatures of syntax::visit mod.
    type E = ();

    impl<'a> Visitor<E> for StrictVersionHashVisitor<'a> {

        fn visit_mac(&mut self, macro: &Mac, e: E) {
            // macro invocations, namely macro_rules definitions,
            // *can* appear as items, even in the expanded crate AST.

            if macro_name(macro).get() == "macro_rules" {
                // Pretty-printing definition to a string strips out
                // surface artifacts (currently), such as the span
                // information, yielding a content-based hash.

                // FIXME (#14132): building temporary string is
                // expensive; a direct content-based hash on token
                // trees might be faster. Implementing this is far
                // easier in short term.
                let macro_defn_as_string =
                    pprust::to_string(|pp_state| pp_state.print_mac(macro));
                macro_defn_as_string.hash(self.st);
            } else {
                // It is not possible to observe any kind of macro
                // invocation at this stage except `macro_rules!`.
                fail!("reached macro somehow: {}",
                      pprust::to_string(|pp_state| pp_state.print_mac(macro)));
            }

            visit::walk_mac(self, macro, e);

            fn macro_name(macro: &Mac) -> token::InternedString {
                match &macro.node {
                    &MacInvocTT(ref path, ref _tts, ref _stx_ctxt) => {
                        let s = path.segments.as_slice();
                        assert_eq!(s.len(), 1);
                        content(s[0].identifier)
                    }
                }
            }
        }

        fn visit_struct_def(&mut self, s: &StructDef, ident: Ident,
                            g: &Generics, _: NodeId, e: E) {
            SawStructDef(content(ident)).hash(self.st);
            visit::walk_generics(self, g, e.clone());
            visit::walk_struct_def(self, s, e)
        }

        fn visit_variant(&mut self, v: &Variant, g: &Generics, e: E) {
            SawVariant.hash(self.st);
            // walk_variant does not call walk_generics, so do it here.
            visit::walk_generics(self, g, e.clone());
            visit::walk_variant(self, v, g, e)
        }

        fn visit_opt_lifetime_ref(&mut self, _: Span, l: &Option<Lifetime>, env: E) {
            SawOptLifetimeRef.hash(self.st);
            // (This is a strange method in the visitor trait, in that
            // it does not expose a walk function to do the subroutine
            // calls.)
            match *l {
                Some(ref l) => self.visit_lifetime_ref(l, env),
                None => ()
            }
        }

        // All of the remaining methods just record (in the hash
        // SipState) that the visitor saw that particular variant
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

        fn visit_ident(&mut self, _: Span, ident: Ident, _: E) {
            SawIdent(content(ident)).hash(self.st);
        }

        fn visit_lifetime_ref(&mut self, l: &Lifetime, _: E) {
            SawLifetimeRef(content(l.name)).hash(self.st);
        }

        fn visit_lifetime_decl(&mut self, l: &LifetimeDef, _: E) {
            SawLifetimeDecl(content(l.lifetime.name)).hash(self.st);
        }

        // We do recursively walk the bodies of functions/methods
        // (rather than omitting their bodies from the hash) since
        // monomorphization and cross-crate inlining generally implies
        // that a change to a crate body will require downstream
        // crates to be recompiled.
        fn visit_expr(&mut self, ex: &Expr, e: E) {
            SawExpr(saw_expr(&ex.node)).hash(self.st); visit::walk_expr(self, ex, e)
        }

        fn visit_stmt(&mut self, s: &Stmt, e: E) {
            SawStmt(saw_stmt(&s.node)).hash(self.st); visit::walk_stmt(self, s, e)
        }

        fn visit_view_item(&mut self, i: &ViewItem, e: E) {
            // Two kinds of view items can affect the ABI for a crate:
            // exported `pub use` view items (since that may expose
            // items that downstream crates can call), and `use
            // foo::Trait`, since changing that may affect method
            // resolution.
            //
            // The simplest approach to handling both of the above is
            // just to adopt the same simple-minded (fine-grained)
            // hash that I am deploying elsewhere here.
            SawViewItem.hash(self.st); visit::walk_view_item(self, i, e)
        }

        fn visit_foreign_item(&mut self, i: &ForeignItem, e: E) {
            // FIXME (#14132) ideally we would incorporate privacy (or
            // perhaps reachability) somewhere here, so foreign items
            // that do not leak into downstream crates would not be
            // part of the ABI.
            SawForeignItem.hash(self.st); visit::walk_foreign_item(self, i, e)
        }

        fn visit_item(&mut self, i: &Item, e: E) {
            // FIXME (#14132) ideally would incorporate reachability
            // analysis somewhere here, so items that never leak into
            // downstream crates (e.g. via monomorphisation or
            // inlining) would not be part of the ABI.
            SawItem.hash(self.st); visit::walk_item(self, i, e)
        }

        fn visit_mod(&mut self, m: &Mod, _s: Span, _n: NodeId, e: E) {
            SawMod.hash(self.st); visit::walk_mod(self, m, e)
        }

        fn visit_decl(&mut self, d: &Decl, e: E) {
            SawDecl.hash(self.st); visit::walk_decl(self, d, e)
        }

        fn visit_ty(&mut self, t: &Ty, e: E) {
            SawTy.hash(self.st); visit::walk_ty(self, t, e)
        }

        fn visit_generics(&mut self, g: &Generics, e: E) {
            SawGenerics.hash(self.st); visit::walk_generics(self, g, e)
        }

        fn visit_fn(&mut self, fk: &FnKind, fd: &FnDecl, b: &Block, s: Span, _: NodeId, e: E) {
            SawFn.hash(self.st); visit::walk_fn(self, fk, fd, b, s, e)
        }

        fn visit_ty_method(&mut self, t: &TypeMethod, e: E) {
            SawTyMethod.hash(self.st); visit::walk_ty_method(self, t, e)
        }

        fn visit_trait_item(&mut self, t: &TraitItem, e: E) {
            SawTraitMethod.hash(self.st); visit::walk_trait_item(self, t, e)
        }

        fn visit_struct_field(&mut self, s: &StructField, e: E) {
            SawStructField.hash(self.st); visit::walk_struct_field(self, s, e)
        }

        fn visit_explicit_self(&mut self, es: &ExplicitSelf, e: E) {
            SawExplicitSelf.hash(self.st); visit::walk_explicit_self(self, es, e)
        }

        fn visit_path(&mut self, path: &Path, _: ast::NodeId, e: E) {
            SawPath.hash(self.st); visit::walk_path(self, path, e)
        }

        fn visit_block(&mut self, b: &Block, e: E) {
            SawBlock.hash(self.st); visit::walk_block(self, b, e)
        }

        fn visit_pat(&mut self, p: &Pat, e: E) {
            SawPat.hash(self.st); visit::walk_pat(self, p, e)
        }

        fn visit_local(&mut self, l: &Local, e: E) {
            SawLocal.hash(self.st); visit::walk_local(self, l, e)
        }

        fn visit_arm(&mut self, a: &Arm, e: E) {
            SawArm.hash(self.st); visit::walk_arm(self, a, e)
        }
    }
}
