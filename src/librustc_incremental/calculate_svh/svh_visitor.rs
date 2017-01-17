// Copyright 2012-2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use self::SawExprComponent::*;
use self::SawAbiComponent::*;
use self::SawItemComponent::*;
use self::SawPatComponent::*;
use self::SawTyComponent::*;
use self::SawTraitOrImplItemComponent::*;
use syntax::abi::Abi;
use syntax::ast::{self, Name, NodeId};
use syntax::attr;
use syntax::parse::token;
use syntax::symbol::{Symbol, InternedString};
use syntax_pos::{Span, NO_EXPANSION, COMMAND_LINE_EXPN, BytePos};
use syntax::tokenstream;
use rustc::hir;
use rustc::hir::*;
use rustc::hir::def::Def;
use rustc::hir::def_id::DefId;
use rustc::hir::intravisit as visit;
use rustc::ty::TyCtxt;
use rustc_data_structures::fnv;
use std::hash::{Hash, Hasher};

use super::def_path_hash::DefPathHashes;
use super::caching_codemap_view::CachingCodemapView;
use super::IchHasher;

const IGNORED_ATTRIBUTES: &'static [&'static str] = &[
    "cfg",
    ::ATTR_IF_THIS_CHANGED,
    ::ATTR_THEN_THIS_WOULD_NEED,
    ::ATTR_DIRTY,
    ::ATTR_CLEAN,
    ::ATTR_DIRTY_METADATA,
    ::ATTR_CLEAN_METADATA
];

pub struct StrictVersionHashVisitor<'a, 'hash: 'a, 'tcx: 'hash> {
    pub tcx: TyCtxt<'hash, 'tcx, 'tcx>,
    pub st: &'a mut IchHasher,
    // collect a deterministic hash of def-ids that we have seen
    def_path_hashes: &'a mut DefPathHashes<'hash, 'tcx>,
    hash_spans: bool,
    codemap: &'a mut CachingCodemapView<'tcx>,
    overflow_checks_enabled: bool,
    hash_bodies: bool,
}

impl<'a, 'hash, 'tcx> StrictVersionHashVisitor<'a, 'hash, 'tcx> {
    pub fn new(st: &'a mut IchHasher,
               tcx: TyCtxt<'hash, 'tcx, 'tcx>,
               def_path_hashes: &'a mut DefPathHashes<'hash, 'tcx>,
               codemap: &'a mut CachingCodemapView<'tcx>,
               hash_spans: bool,
               hash_bodies: bool)
               -> Self {
        let check_overflow = tcx.sess.opts.debugging_opts.force_overflow_checks
            .unwrap_or(tcx.sess.opts.debug_assertions);

        StrictVersionHashVisitor {
            st: st,
            tcx: tcx,
            def_path_hashes: def_path_hashes,
            hash_spans: hash_spans,
            codemap: codemap,
            overflow_checks_enabled: check_overflow,
            hash_bodies: hash_bodies,
        }
    }

    fn compute_def_id_hash(&mut self, def_id: DefId) -> u64 {
        self.def_path_hashes.hash(def_id)
    }

    // Hash a span in a stable way. We can't directly hash the span's BytePos
    // fields (that would be similar to hashing pointers, since those are just
    // offsets into the CodeMap). Instead, we hash the (file name, line, column)
    // triple, which stays the same even if the containing FileMap has moved
    // within the CodeMap.
    // Also note that we are hashing byte offsets for the column, not unicode
    // codepoint offsets. For the purpose of the hash that's sufficient.
    // Also, hashing filenames is expensive so we avoid doing it twice when the
    // span starts and ends in the same file, which is almost always the case.
    fn hash_span(&mut self, span: Span) {
        debug!("hash_span: st={:?}", self.st);

        // If this is not an empty or invalid span, we want to hash the last
        // position that belongs to it, as opposed to hashing the first
        // position past it.
        let span_hi = if span.hi > span.lo {
            // We might end up in the middle of a multibyte character here,
            // but that's OK, since we are not trying to decode anything at
            // this position.
            span.hi - BytePos(1)
        } else {
            span.hi
        };

        let expn_kind = match span.expn_id {
            NO_EXPANSION => SawSpanExpnKind::NoExpansion,
            COMMAND_LINE_EXPN => SawSpanExpnKind::CommandLine,
            _ => SawSpanExpnKind::SomeExpansion,
        };

        let loc1 = self.codemap.byte_pos_to_line_and_col(span.lo);
        let loc1 = loc1.as_ref()
                       .map(|&(ref fm, line, col)| (&fm.name[..], line, col))
                       .unwrap_or(("???", 0, BytePos(0)));

        let loc2 = self.codemap.byte_pos_to_line_and_col(span_hi);
        let loc2 = loc2.as_ref()
                       .map(|&(ref fm, line, col)| (&fm.name[..], line, col))
                       .unwrap_or(("???", 0, BytePos(0)));

        let saw = if loc1.0 == loc2.0 {
            SawSpan(loc1.0,
                    loc1.1, loc1.2,
                    loc2.1, loc2.2,
                    expn_kind)
        } else {
            SawSpanTwoFiles(loc1.0, loc1.1, loc1.2,
                            loc2.0, loc2.1, loc2.2,
                            expn_kind)
        };
        saw.hash(self.st);

        if expn_kind == SawSpanExpnKind::SomeExpansion {
            let call_site = self.codemap.codemap().source_callsite(span);
            self.hash_span(call_site);
        }
    }

    fn hash_discriminant<T>(&mut self, v: &T) {
        unsafe {
            let disr = ::std::intrinsics::discriminant_value(v);
            debug!("hash_discriminant: disr={}, st={:?}", disr, self.st);
            disr.hash(self.st);
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
    SawIdent(InternedString),
    SawStructDef(InternedString),

    SawLifetime,
    SawLifetimeDef(usize),

    SawMod,
    SawForeignItem(SawForeignItemComponent),
    SawItem(SawItemComponent),
    SawTy(SawTyComponent),
    SawFnDecl(bool),
    SawGenerics,
    SawTraitItem(SawTraitOrImplItemComponent),
    SawImplItem(SawTraitOrImplItemComponent),
    SawStructField,
    SawVariant(bool),
    SawQPath,
    SawPathSegment,
    SawPathParameters,
    SawBlock,
    SawPat(SawPatComponent),
    SawLocal,
    SawArm,
    SawExpr(SawExprComponent<'a>),
    SawStmt,
    SawVis,
    SawAssociatedItemKind(hir::AssociatedItemKind),
    SawDefaultness(hir::Defaultness),
    SawWherePredicate,
    SawTyParamBound,
    SawPolyTraitRef,
    SawAssocTypeBinding,
    SawAttribute(ast::AttrStyle),
    SawMacroDef,
    SawSpan(&'a str,
            usize, BytePos,
            usize, BytePos,
            SawSpanExpnKind),
    SawSpanTwoFiles(&'a str, usize, BytePos,
                    &'a str, usize, BytePos,
                    SawSpanExpnKind),
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
///
/// The xxxComponent enums and saw_xxx functions for Item, Pat,
/// Ty, TraitItem and ImplItem follow the same methodology.
#[derive(Hash)]
enum SawExprComponent<'a> {

    SawExprLoop(Option<InternedString>),
    SawExprField(InternedString),
    SawExprTupField(usize),
    SawExprBreak(Option<InternedString>),
    SawExprAgain(Option<InternedString>),

    SawExprBox,
    SawExprArray,
    SawExprCall,
    SawExprMethodCall,
    SawExprTup,
    SawExprBinary(hir::BinOp_),
    SawExprUnary(hir::UnOp),
    SawExprLit(ast::LitKind),
    SawExprLitStr(InternedString, ast::StrStyle),
    SawExprLitFloat(InternedString, Option<ast::FloatTy>),
    SawExprCast,
    SawExprType,
    SawExprIf,
    SawExprWhile,
    SawExprMatch,
    SawExprClosure(CaptureClause),
    SawExprBlock,
    SawExprAssign,
    SawExprAssignOp(hir::BinOp_),
    SawExprIndex,
    SawExprPath,
    SawExprAddrOf(hir::Mutability),
    SawExprRet,
    SawExprInlineAsm(StableInlineAsm<'a>),
    SawExprStruct,
    SawExprRepeat,
}

// The boolean returned indicates whether the span of this expression is always
// significant, regardless of debuginfo.
fn saw_expr<'a>(node: &'a Expr_,
                overflow_checks_enabled: bool)
                -> (SawExprComponent<'a>, bool) {
    let binop_can_panic_at_runtime = |binop| {
        match binop {
            BiAdd |
            BiSub |
            BiMul => overflow_checks_enabled,

            BiDiv |
            BiRem => true,

            BiAnd |
            BiOr |
            BiBitXor |
            BiBitAnd |
            BiBitOr |
            BiShl |
            BiShr |
            BiEq |
            BiLt |
            BiLe |
            BiNe |
            BiGe |
            BiGt => false
        }
    };

    let unop_can_panic_at_runtime = |unop| {
        match unop {
            UnDeref |
            UnNot => false,
            UnNeg => overflow_checks_enabled,
        }
    };

    match *node {
        ExprBox(..)              => (SawExprBox, false),
        ExprArray(..)            => (SawExprArray, false),
        ExprCall(..)             => (SawExprCall, false),
        ExprMethodCall(..)       => (SawExprMethodCall, false),
        ExprTup(..)              => (SawExprTup, false),
        ExprBinary(op, ..)       => {
            (SawExprBinary(op.node), binop_can_panic_at_runtime(op.node))
        }
        ExprUnary(op, _)         => {
            (SawExprUnary(op), unop_can_panic_at_runtime(op))
        }
        ExprLit(ref lit)         => (saw_lit(lit), false),
        ExprCast(..)             => (SawExprCast, false),
        ExprType(..)             => (SawExprType, false),
        ExprIf(..)               => (SawExprIf, false),
        ExprWhile(..)            => (SawExprWhile, false),
        ExprLoop(_, id, _)       => (SawExprLoop(id.map(|id| id.node.as_str())), false),
        ExprMatch(..)            => (SawExprMatch, false),
        ExprClosure(cc, _, _, _) => (SawExprClosure(cc), false),
        ExprBlock(..)            => (SawExprBlock, false),
        ExprAssign(..)           => (SawExprAssign, false),
        ExprAssignOp(op, ..)     => {
            (SawExprAssignOp(op.node), binop_can_panic_at_runtime(op.node))
        }
        ExprField(_, name)       => (SawExprField(name.node.as_str()), false),
        ExprTupField(_, id)      => (SawExprTupField(id.node), false),
        ExprIndex(..)            => (SawExprIndex, true),
        ExprPath(_)              => (SawExprPath, false),
        ExprAddrOf(m, _)         => (SawExprAddrOf(m), false),
        ExprBreak(label, _)      => (SawExprBreak(label.map(|l| l.name.as_str())), false),
        ExprAgain(label)         => (SawExprAgain(label.map(|l| l.name.as_str())), false),
        ExprRet(..)              => (SawExprRet, false),
        ExprInlineAsm(ref a,..)  => (SawExprInlineAsm(StableInlineAsm(a)), false),
        ExprStruct(..)           => (SawExprStruct, false),
        ExprRepeat(..)           => (SawExprRepeat, false),
    }
}

fn saw_lit(lit: &ast::Lit) -> SawExprComponent<'static> {
    match lit.node {
        ast::LitKind::Str(s, style) => SawExprLitStr(s.as_str(), style),
        ast::LitKind::Float(s, ty) => SawExprLitFloat(s.as_str(), Some(ty)),
        ast::LitKind::FloatUnsuffixed(s) => SawExprLitFloat(s.as_str(), None),
        ref node @ _ => SawExprLit(node.clone()),
    }
}

#[derive(Hash)]
enum SawItemComponent {
    SawItemExternCrate,
    SawItemUse(UseKind),
    SawItemStatic(Mutability),
    SawItemConst,
    SawItemFn(Unsafety, Constness, Abi),
    SawItemMod,
    SawItemForeignMod(Abi),
    SawItemTy,
    SawItemEnum,
    SawItemStruct,
    SawItemUnion,
    SawItemTrait(Unsafety),
    SawItemDefaultImpl(Unsafety),
    SawItemImpl(Unsafety, ImplPolarity)
}

fn saw_item(node: &Item_) -> SawItemComponent {
    match *node {
        ItemExternCrate(..) => SawItemExternCrate,
        ItemUse(_, kind) => SawItemUse(kind),
        ItemStatic(_, mutability, _) => SawItemStatic(mutability),
        ItemConst(..) =>SawItemConst,
        ItemFn(_, unsafety, constness, abi, _, _) => SawItemFn(unsafety, constness, abi),
        ItemMod(..) => SawItemMod,
        ItemForeignMod(ref fm) => SawItemForeignMod(fm.abi),
        ItemTy(..) => SawItemTy,
        ItemEnum(..) => SawItemEnum,
        ItemStruct(..) => SawItemStruct,
        ItemUnion(..) => SawItemUnion,
        ItemTrait(unsafety, ..) => SawItemTrait(unsafety),
        ItemDefaultImpl(unsafety, _) => SawItemDefaultImpl(unsafety),
        ItemImpl(unsafety, implpolarity, ..) => SawItemImpl(unsafety, implpolarity)
    }
}

#[derive(Hash)]
enum SawForeignItemComponent {
    Static { mutable: bool },
    Fn,
}

#[derive(Hash)]
enum SawPatComponent {
    SawPatWild,
    SawPatBinding(BindingMode),
    SawPatStruct,
    SawPatTupleStruct,
    SawPatPath,
    SawPatTuple,
    SawPatBox,
    SawPatRef(Mutability),
    SawPatLit,
    SawPatRange,
    SawPatSlice
}

fn saw_pat(node: &PatKind) -> SawPatComponent {
    match *node {
        PatKind::Wild => SawPatWild,
        PatKind::Binding(bindingmode, ..) => SawPatBinding(bindingmode),
        PatKind::Struct(..) => SawPatStruct,
        PatKind::TupleStruct(..) => SawPatTupleStruct,
        PatKind::Path(_) => SawPatPath,
        PatKind::Tuple(..) => SawPatTuple,
        PatKind::Box(..) => SawPatBox,
        PatKind::Ref(_, mutability) => SawPatRef(mutability),
        PatKind::Lit(..) => SawPatLit,
        PatKind::Range(..) => SawPatRange,
        PatKind::Slice(..) => SawPatSlice
    }
}

#[derive(Hash)]
enum SawTyComponent {
    SawTySlice,
    SawTyArray,
    SawTyPtr(Mutability),
    SawTyRptr(Mutability),
    SawTyBareFn(Unsafety, Abi),
    SawTyNever,
    SawTyTup,
    SawTyPath,
    SawTyObjectSum,
    SawTyImplTrait,
    SawTyTypeof,
    SawTyInfer
}

fn saw_ty(node: &Ty_) -> SawTyComponent {
    match *node {
      TySlice(..) => SawTySlice,
      TyArray(..) => SawTyArray,
      TyPtr(ref mty) => SawTyPtr(mty.mutbl),
      TyRptr(_, ref mty) => SawTyRptr(mty.mutbl),
      TyBareFn(ref barefnty) => SawTyBareFn(barefnty.unsafety, barefnty.abi),
      TyNever => SawTyNever,
      TyTup(..) => SawTyTup,
      TyPath(_) => SawTyPath,
      TyTraitObject(..) => SawTyObjectSum,
      TyImplTrait(..) => SawTyImplTrait,
      TyTypeof(..) => SawTyTypeof,
      TyInfer => SawTyInfer
    }
}

#[derive(Hash)]
enum SawTraitOrImplItemComponent {
    SawTraitOrImplItemConst,
    // The boolean signifies whether a body is present
    SawTraitOrImplItemMethod(Unsafety, Constness, Abi, bool),
    SawTraitOrImplItemType
}

fn saw_trait_item(ti: &TraitItemKind) -> SawTraitOrImplItemComponent {
    match *ti {
        TraitItemKind::Const(..) => SawTraitOrImplItemConst,
        TraitItemKind::Method(ref sig, TraitMethod::Required(_)) =>
            SawTraitOrImplItemMethod(sig.unsafety, sig.constness, sig.abi, false),
        TraitItemKind::Method(ref sig, TraitMethod::Provided(_)) =>
            SawTraitOrImplItemMethod(sig.unsafety, sig.constness, sig.abi, true),
        TraitItemKind::Type(..) => SawTraitOrImplItemType
    }
}

fn saw_impl_item(ii: &ImplItemKind) -> SawTraitOrImplItemComponent {
    match *ii {
        ImplItemKind::Const(..) => SawTraitOrImplItemConst,
        ImplItemKind::Method(ref sig, _) =>
            SawTraitOrImplItemMethod(sig.unsafety, sig.constness, sig.abi, true),
        ImplItemKind::Type(..) => SawTraitOrImplItemType
    }
}

#[derive(Clone, Copy, Hash, Eq, PartialEq)]
enum SawSpanExpnKind {
    NoExpansion,
    CommandLine,
    SomeExpansion,
}

/// A wrapper that provides a stable Hash implementation.
struct StableInlineAsm<'a>(&'a InlineAsm);

impl<'a> Hash for StableInlineAsm<'a> {
    fn hash<H: Hasher>(&self, state: &mut H) {
        let InlineAsm {
            asm,
            asm_str_style,
            ref outputs,
            ref inputs,
            ref clobbers,
            volatile,
            alignstack,
            dialect,
            expn_id: _, // This is used for error reporting
        } = *self.0;

        asm.as_str().hash(state);
        asm_str_style.hash(state);
        outputs.len().hash(state);
        for output in outputs {
            let InlineAsmOutput { constraint, is_rw, is_indirect } = *output;
            constraint.as_str().hash(state);
            is_rw.hash(state);
            is_indirect.hash(state);
        }
        inputs.len().hash(state);
        for input in inputs {
            input.as_str().hash(state);
        }
        clobbers.len().hash(state);
        for clobber in clobbers {
            clobber.as_str().hash(state);
        }
        volatile.hash(state);
        alignstack.hash(state);
        dialect.hash(state);
    }
}

macro_rules! hash_attrs {
    ($visitor:expr, $attrs:expr) => ({
        let attrs = $attrs;
        if attrs.len() > 0 {
            $visitor.hash_attributes(attrs);
        }
    })
}

macro_rules! hash_span {
    ($visitor:expr, $span:expr) => ({
        hash_span!($visitor, $span, false)
    });
    ($visitor:expr, $span:expr, $force:expr) => ({
        if $force || $visitor.hash_spans {
            $visitor.hash_span($span);
        }
    });
}

impl<'a, 'hash, 'tcx> visit::Visitor<'tcx> for StrictVersionHashVisitor<'a, 'hash, 'tcx> {
    fn nested_visit_map<'this>(&'this mut self) -> visit::NestedVisitorMap<'this, 'tcx> {
        if self.hash_bodies {
            visit::NestedVisitorMap::OnlyBodies(&self.tcx.map)
        } else {
            visit::NestedVisitorMap::None
        }
    }

    fn visit_variant_data(&mut self,
                          s: &'tcx VariantData,
                          name: Name,
                          _: &'tcx Generics,
                          _: NodeId,
                          span: Span) {
        debug!("visit_variant_data: st={:?}", self.st);
        SawStructDef(name.as_str()).hash(self.st);
        hash_span!(self, span);
        visit::walk_struct_def(self, s);
    }

    fn visit_variant(&mut self,
                     v: &'tcx Variant,
                     g: &'tcx Generics,
                     item_id: NodeId) {
        debug!("visit_variant: st={:?}", self.st);
        SawVariant(v.node.disr_expr.is_some()).hash(self.st);
        hash_attrs!(self, &v.node.attrs);
        visit::walk_variant(self, v, g, item_id)
    }

    fn visit_name(&mut self, span: Span, name: Name) {
        debug!("visit_name: st={:?}", self.st);
        SawIdent(name.as_str()).hash(self.st);
        hash_span!(self, span);
    }

    fn visit_lifetime(&mut self, l: &'tcx Lifetime) {
        debug!("visit_lifetime: st={:?}", self.st);
        SawLifetime.hash(self.st);
        visit::walk_lifetime(self, l);
    }

    fn visit_lifetime_def(&mut self, l: &'tcx LifetimeDef) {
        debug!("visit_lifetime_def: st={:?}", self.st);
        SawLifetimeDef(l.bounds.len()).hash(self.st);
        visit::walk_lifetime_def(self, l);
    }

    fn visit_expr(&mut self, ex: &'tcx Expr) {
        debug!("visit_expr: st={:?}", self.st);
        let (saw_expr, force_span) = saw_expr(&ex.node,
                                              self.overflow_checks_enabled);
        SawExpr(saw_expr).hash(self.st);
        // No need to explicitly hash the discriminant here, since we are
        // implicitly hashing the discriminant of SawExprComponent.
        hash_span!(self, ex.span, force_span);
        hash_attrs!(self, &ex.attrs);

        // Always hash nested constant bodies (e.g. n in `[x; n]`).
        let hash_bodies = self.hash_bodies;
        self.hash_bodies = true;
        visit::walk_expr(self, ex);
        self.hash_bodies = hash_bodies;
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
            StmtExpr(..) => {
                SawStmt.hash(self.st);
                self.hash_discriminant(&s.node);
                hash_span!(self, s.span);
            }
            StmtSemi(..) => {
                SawStmt.hash(self.st);
                self.hash_discriminant(&s.node);
                hash_span!(self, s.span);
            }
        }

        visit::walk_stmt(self, s)
    }

    fn visit_foreign_item(&mut self, i: &'tcx ForeignItem) {
        debug!("visit_foreign_item: st={:?}", self.st);

        match i.node {
            ForeignItemFn(..) => {
                SawForeignItem(SawForeignItemComponent::Fn)
            }
            ForeignItemStatic(_, mutable) => {
                SawForeignItem(SawForeignItemComponent::Static {
                    mutable: mutable
                })
            }
        }.hash(self.st);

        hash_span!(self, i.span);
        hash_attrs!(self, &i.attrs);
        visit::walk_foreign_item(self, i)
    }

    fn visit_item(&mut self, i: &'tcx Item) {
        debug!("visit_item: {:?} st={:?}", i, self.st);

        self.maybe_enable_overflow_checks(&i.attrs);

        SawItem(saw_item(&i.node)).hash(self.st);
        hash_span!(self, i.span);
        hash_attrs!(self, &i.attrs);
        visit::walk_item(self, i)
    }

    fn visit_mod(&mut self, m: &'tcx Mod, span: Span, n: NodeId) {
        debug!("visit_mod: st={:?}", self.st);
        SawMod.hash(self.st);
        hash_span!(self, span);
        visit::walk_mod(self, m, n)
    }

    fn visit_ty(&mut self, t: &'tcx Ty) {
        debug!("visit_ty: st={:?}", self.st);
        SawTy(saw_ty(&t.node)).hash(self.st);
        hash_span!(self, t.span);

        // Always hash nested constant bodies (e.g. N in `[T; N]`).
        let hash_bodies = self.hash_bodies;
        self.hash_bodies = true;
        visit::walk_ty(self, t);
        self.hash_bodies = hash_bodies;
    }

    fn visit_generics(&mut self, g: &'tcx Generics) {
        debug!("visit_generics: st={:?}", self.st);
        SawGenerics.hash(self.st);
        visit::walk_generics(self, g)
    }

    fn visit_fn_decl(&mut self, fd: &'tcx FnDecl) {
        debug!("visit_fn_decl: st={:?}", self.st);
        SawFnDecl(fd.variadic).hash(self.st);
        visit::walk_fn_decl(self, fd)
    }

    fn visit_trait_item(&mut self, ti: &'tcx TraitItem) {
        debug!("visit_trait_item: st={:?}", self.st);

        self.maybe_enable_overflow_checks(&ti.attrs);

        SawTraitItem(saw_trait_item(&ti.node)).hash(self.st);
        hash_span!(self, ti.span);
        hash_attrs!(self, &ti.attrs);
        visit::walk_trait_item(self, ti)
    }

    fn visit_impl_item(&mut self, ii: &'tcx ImplItem) {
        debug!("visit_impl_item: st={:?}", self.st);

        self.maybe_enable_overflow_checks(&ii.attrs);

        SawImplItem(saw_impl_item(&ii.node)).hash(self.st);
        hash_span!(self, ii.span);
        hash_attrs!(self, &ii.attrs);
        visit::walk_impl_item(self, ii)
    }

    fn visit_struct_field(&mut self, s: &'tcx StructField) {
        debug!("visit_struct_field: st={:?}", self.st);
        SawStructField.hash(self.st);
        hash_span!(self, s.span);
        hash_attrs!(self, &s.attrs);
        visit::walk_struct_field(self, s)
    }

    fn visit_qpath(&mut self, qpath: &'tcx QPath, id: NodeId, span: Span) {
        debug!("visit_qpath: st={:?}", self.st);
        SawQPath.hash(self.st);
        self.hash_discriminant(qpath);
        visit::walk_qpath(self, qpath, id, span)
    }

    fn visit_path(&mut self, path: &'tcx Path, _: ast::NodeId) {
        debug!("visit_path: st={:?}", self.st);
        hash_span!(self, path.span);
        visit::walk_path(self, path)
    }

    fn visit_def_mention(&mut self, def: Def) {
        self.hash_def(def);
    }

    fn visit_block(&mut self, b: &'tcx Block) {
        debug!("visit_block: st={:?}", self.st);
        SawBlock.hash(self.st);
        hash_span!(self, b.span);
        visit::walk_block(self, b)
    }

    fn visit_pat(&mut self, p: &'tcx Pat) {
        debug!("visit_pat: st={:?}", self.st);
        SawPat(saw_pat(&p.node)).hash(self.st);
        hash_span!(self, p.span);
        visit::walk_pat(self, p)
    }

    fn visit_local(&mut self, l: &'tcx Local) {
        debug!("visit_local: st={:?}", self.st);
        SawLocal.hash(self.st);
        hash_attrs!(self, &l.attrs);
        visit::walk_local(self, l)
        // No need to hash span, we are hashing all component spans
    }

    fn visit_arm(&mut self, a: &'tcx Arm) {
        debug!("visit_arm: st={:?}", self.st);
        SawArm.hash(self.st);
        hash_attrs!(self, &a.attrs);
        visit::walk_arm(self, a)
    }

    fn visit_id(&mut self, id: NodeId) {
        debug!("visit_id: id={} st={:?}", id, self.st);
        self.hash_resolve(id)
    }

    fn visit_vis(&mut self, v: &'tcx Visibility) {
        debug!("visit_vis: st={:?}", self.st);
        SawVis.hash(self.st);
        self.hash_discriminant(v);
        visit::walk_vis(self, v)
    }

    fn visit_associated_item_kind(&mut self, kind: &'tcx AssociatedItemKind) {
        debug!("visit_associated_item_kind: st={:?}", self.st);
        SawAssociatedItemKind(*kind).hash(self.st);
        visit::walk_associated_item_kind(self, kind);
    }

    fn visit_defaultness(&mut self, defaultness: &'tcx Defaultness) {
        debug!("visit_associated_item_kind: st={:?}", self.st);
        SawDefaultness(*defaultness).hash(self.st);
        visit::walk_defaultness(self, defaultness);
    }

    fn visit_where_predicate(&mut self, predicate: &'tcx WherePredicate) {
        debug!("visit_where_predicate: st={:?}", self.st);
        SawWherePredicate.hash(self.st);
        self.hash_discriminant(predicate);
        // Ignoring span. Any important nested components should be visited.
        visit::walk_where_predicate(self, predicate)
    }

    fn visit_ty_param_bound(&mut self, bounds: &'tcx TyParamBound) {
        debug!("visit_ty_param_bound: st={:?}", self.st);
        SawTyParamBound.hash(self.st);
        self.hash_discriminant(bounds);
        // The TraitBoundModifier in TraitTyParamBound will be hash in
        // visit_poly_trait_ref()
        visit::walk_ty_param_bound(self, bounds)
    }

    fn visit_poly_trait_ref(&mut self, t: &'tcx PolyTraitRef, m: &'tcx TraitBoundModifier) {
        debug!("visit_poly_trait_ref: st={:?}", self.st);
        SawPolyTraitRef.hash(self.st);
        m.hash(self.st);
        visit::walk_poly_trait_ref(self, t, m)
    }

    fn visit_path_segment(&mut self, path_span: Span, path_segment: &'tcx PathSegment) {
        debug!("visit_path_segment: st={:?}", self.st);
        SawPathSegment.hash(self.st);
        visit::walk_path_segment(self, path_span, path_segment)
    }

    fn visit_path_parameters(&mut self, path_span: Span, path_parameters: &'tcx PathParameters) {
        debug!("visit_path_parameters: st={:?}", self.st);
        SawPathParameters.hash(self.st);
        self.hash_discriminant(path_parameters);
        visit::walk_path_parameters(self, path_span, path_parameters)
    }

    fn visit_assoc_type_binding(&mut self, type_binding: &'tcx TypeBinding) {
        debug!("visit_assoc_type_binding: st={:?}", self.st);
        SawAssocTypeBinding.hash(self.st);
        hash_span!(self, type_binding.span);
        visit::walk_assoc_type_binding(self, type_binding)
    }

    fn visit_attribute(&mut self, _: &ast::Attribute) {
        // We explicitly do not use this method, since doing that would
        // implicitly impose an order on the attributes being hashed, while we
        // explicitly don't want their order to matter
    }

    fn visit_macro_def(&mut self, macro_def: &'tcx MacroDef) {
        debug!("visit_macro_def: st={:?}", self.st);
        SawMacroDef.hash(self.st);
        hash_attrs!(self, &macro_def.attrs);
        for tt in &macro_def.body {
            self.hash_token_tree(tt);
        }
        visit::walk_macro_def(self, macro_def)
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

    fn hash_def(&mut self, def: Def) {
        match def {
            // Crucial point: for all of these variants, the variant +
            // add'l data that is added is always the same if the
            // def-id is the same, so it suffices to hash the def-id
            Def::Fn(..) |
            Def::Mod(..) |
            Def::Static(..) |
            Def::Variant(..) |
            Def::VariantCtor(..) |
            Def::Enum(..) |
            Def::TyAlias(..) |
            Def::AssociatedTy(..) |
            Def::TyParam(..) |
            Def::Struct(..) |
            Def::StructCtor(..) |
            Def::Union(..) |
            Def::Trait(..) |
            Def::Method(..) |
            Def::Const(..) |
            Def::AssociatedConst(..) |
            Def::Local(..) |
            Def::Upvar(..) |
            Def::Macro(..) => {
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

    fn hash_meta_item(&mut self, meta_item: &ast::MetaItem) {
        debug!("hash_meta_item: st={:?}", self.st);

        // ignoring span information, it doesn't matter here
        self.hash_discriminant(&meta_item.node);
        meta_item.name.as_str().len().hash(self.st);
        meta_item.name.as_str().hash(self.st);

        match meta_item.node {
            ast::MetaItemKind::Word => {}
            ast::MetaItemKind::NameValue(ref lit) => saw_lit(lit).hash(self.st),
            ast::MetaItemKind::List(ref items) => {
                // Sort subitems so the hash does not depend on their order
                let indices = self.indices_sorted_by(&items, |p| {
                    (p.name().map(Symbol::as_str), fnv::hash(&p.literal().map(saw_lit)))
                });
                items.len().hash(self.st);
                for (index, &item_index) in indices.iter().enumerate() {
                    index.hash(self.st);
                    let nested_meta_item: &ast::NestedMetaItemKind = &items[item_index].node;
                    self.hash_discriminant(nested_meta_item);
                    match *nested_meta_item {
                        ast::NestedMetaItemKind::MetaItem(ref meta_item) => {
                            self.hash_meta_item(meta_item);
                        }
                        ast::NestedMetaItemKind::Literal(ref lit) => {
                            saw_lit(lit).hash(self.st);
                        }
                    }
                }
            }
        }
    }

    pub fn hash_attributes(&mut self, attributes: &[ast::Attribute]) {
        debug!("hash_attributes: st={:?}", self.st);
        let indices = self.indices_sorted_by(attributes, |attr| attr.name());

        for i in indices {
            let attr = &attributes[i];
            if !attr.is_sugared_doc &&
               !IGNORED_ATTRIBUTES.contains(&&*attr.value.name().as_str()) {
                SawAttribute(attr.style).hash(self.st);
                self.hash_meta_item(&attr.value);
            }
        }
    }

    fn indices_sorted_by<T, K, F>(&mut self, items: &[T], get_key: F) -> Vec<usize>
        where K: Ord,
              F: Fn(&T) -> K
    {
        let mut indices = Vec::with_capacity(items.len());
        indices.extend(0 .. items.len());
        indices.sort_by_key(|index| get_key(&items[*index]));
        indices
    }

    fn maybe_enable_overflow_checks(&mut self, item_attrs: &[ast::Attribute]) {
        if attr::contains_name(item_attrs, "rustc_inherit_overflow_checks") {
            self.overflow_checks_enabled = true;
        }
    }

    fn hash_token_tree(&mut self, tt: &tokenstream::TokenTree) {
        self.hash_discriminant(tt);
        match *tt {
            tokenstream::TokenTree::Token(span, ref token) => {
                hash_span!(self, span);
                self.hash_token(token, span);
            }
            tokenstream::TokenTree::Delimited(span, ref delimited) => {
                hash_span!(self, span);
                let tokenstream::Delimited {
                    ref delim,
                    open_span,
                    ref tts,
                    close_span,
                } = **delimited;

                delim.hash(self.st);
                hash_span!(self, open_span);
                tts.len().hash(self.st);
                for sub_tt in tts {
                    self.hash_token_tree(sub_tt);
                }
                hash_span!(self, close_span);
            }
            tokenstream::TokenTree::Sequence(span, ref sequence_repetition) => {
                hash_span!(self, span);
                let tokenstream::SequenceRepetition {
                    ref tts,
                    ref separator,
                    op,
                    num_captures,
                } = **sequence_repetition;

                tts.len().hash(self.st);
                for sub_tt in tts {
                    self.hash_token_tree(sub_tt);
                }
                self.hash_discriminant(separator);
                if let Some(ref separator) = *separator {
                    self.hash_token(separator, span);
                }
                op.hash(self.st);
                num_captures.hash(self.st);
            }
        }
    }

    fn hash_token(&mut self,
                  token: &token::Token,
                  error_reporting_span: Span) {
        self.hash_discriminant(token);
        match *token {
            token::Token::Eq |
            token::Token::Lt |
            token::Token::Le |
            token::Token::EqEq |
            token::Token::Ne |
            token::Token::Ge |
            token::Token::Gt |
            token::Token::AndAnd |
            token::Token::OrOr |
            token::Token::Not |
            token::Token::Tilde |
            token::Token::At |
            token::Token::Dot |
            token::Token::DotDot |
            token::Token::DotDotDot |
            token::Token::Comma |
            token::Token::Semi |
            token::Token::Colon |
            token::Token::ModSep |
            token::Token::RArrow |
            token::Token::LArrow |
            token::Token::FatArrow |
            token::Token::Pound |
            token::Token::Dollar |
            token::Token::Question |
            token::Token::Underscore |
            token::Token::Whitespace |
            token::Token::Comment |
            token::Token::Eof => {}

            token::Token::BinOp(bin_op_token) |
            token::Token::BinOpEq(bin_op_token) => bin_op_token.hash(self.st),

            token::Token::OpenDelim(delim_token) |
            token::Token::CloseDelim(delim_token) => delim_token.hash(self.st),

            token::Token::Literal(ref lit, ref opt_name) => {
                self.hash_discriminant(lit);
                match *lit {
                    token::Lit::Byte(val) |
                    token::Lit::Char(val) |
                    token::Lit::Integer(val) |
                    token::Lit::Float(val) |
                    token::Lit::Str_(val) |
                    token::Lit::ByteStr(val) => val.as_str().hash(self.st),
                    token::Lit::StrRaw(val, n) |
                    token::Lit::ByteStrRaw(val, n) => {
                        val.as_str().hash(self.st);
                        n.hash(self.st);
                    }
                };
                opt_name.map(ast::Name::as_str).hash(self.st);
            }

            token::Token::Ident(ident) |
            token::Token::Lifetime(ident) |
            token::Token::SubstNt(ident) => ident.name.as_str().hash(self.st),
            token::Token::MatchNt(ident1, ident2) => {
                ident1.name.as_str().hash(self.st);
                ident2.name.as_str().hash(self.st);
            }

            token::Token::Interpolated(ref non_terminal) => {
                // FIXME(mw): This could be implemented properly. It's just a
                //            lot of work, since we would need to hash the AST
                //            in a stable way, in addition to the HIR.
                //            Since this is hardly used anywhere, just emit a
                //            warning for now.
                if self.tcx.sess.opts.debugging_opts.incremental.is_some() {
                    let msg = format!("Quasi-quoting might make incremental \
                                       compilation very inefficient: {:?}",
                                      non_terminal);
                    self.tcx.sess.span_warn(error_reporting_span, &msg[..]);
                }

                non_terminal.hash(self.st);
            }

            token::Token::DocComment(val) |
            token::Token::Shebang(val) => val.as_str().hash(self.st),
        }
    }

    pub fn hash_crate_root_module(&mut self, krate: &'tcx Crate) {
        let hir::Crate {
            ref module,
            ref attrs,
            span,

            // These fields are handled separately:
            exported_macros: _,
            items: _,
            trait_items: _,
            impl_items: _,
            bodies: _,
        } = *krate;

        visit::Visitor::visit_mod(self, module, span, ast::CRATE_NODE_ID);
        // Crate attributes are not copied over to the root `Mod`, so hash them
        // explicitly here.
        hash_attrs!(self, attrs);
    }
}
