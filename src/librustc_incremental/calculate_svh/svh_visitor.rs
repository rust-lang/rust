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

use self::SawExprComponent::*;
use self::SawAbiComponent::*;
use syntax::ast::{self, Name, NodeId, Attribute};
use syntax::parse::token;
use syntax::codemap::CodeMap;
use syntax_pos::{Span, NO_EXPANSION, COMMAND_LINE_EXPN, BytePos, FileMap};
use rustc::hir;
use rustc::hir::*;
use rustc::hir::def::{Def, PathResolution};
use rustc::hir::def_id::DefId;
use rustc::hir::intravisit as visit;
use rustc::ty::TyCtxt;
use std::rc::Rc;
use std::hash::{Hash, SipHasher};

use super::def_path_hash::DefPathHashes;

const IGNORED_ATTRIBUTES: &'static [&'static str] = &["cfg",
                                                      "rustc_clean",
                                                      "rustc_dirty"];

pub struct StrictVersionHashVisitor<'a, 'hash: 'a, 'tcx: 'hash> {
    pub tcx: TyCtxt<'hash, 'tcx, 'tcx>,
    pub st: &'a mut SipHasher,
    // collect a deterministic hash of def-ids that we have seen
    def_path_hashes: &'a mut DefPathHashes<'hash, 'tcx>,
    hash_spans: bool,
    codemap: CachingCodemapView<'tcx>,
}

struct CachingCodemapView<'tcx> {
    codemap: &'tcx CodeMap,
    // Format: (line number, line-start, line-end, file)
    line_cache: [(usize, BytePos, BytePos, Rc<FileMap>); 4],
    eviction_index: usize,
}

impl<'tcx> CachingCodemapView<'tcx> {
    fn new<'a>(tcx: TyCtxt<'a, 'tcx, 'tcx>) -> CachingCodemapView<'tcx> {
        let codemap = tcx.sess.codemap();
        let first_file = codemap.files.borrow()[0].clone();

        CachingCodemapView {
            codemap: codemap,
            line_cache: [(0, BytePos(0), BytePos(0), first_file.clone()),
                         (0, BytePos(0), BytePos(0), first_file.clone()),
                         (0, BytePos(0), BytePos(0), first_file.clone()),
                         (0, BytePos(0), BytePos(0), first_file.clone())],
            eviction_index: 0,
        }
    }

    fn byte_pos_to_line_and_col(&mut self,
                                pos: BytePos)
                                -> (Rc<FileMap>, usize, BytePos) {
        // Check if the position is in one of the cached lines
        for &(line, start, end, ref file) in self.line_cache.iter() {
            if pos >= start && pos < end {
                return (file.clone(), line, pos - start);
            }
        }

        // Check whether we have a cached line in the correct file, so we can
        // overwrite it without having to look up the file again.
        for &mut (ref mut line,
                  ref mut start,
                  ref mut end,
                  ref file) in self.line_cache.iter_mut() {
            if pos >= file.start_pos && pos < file.end_pos {
                let line_index = file.lookup_line(pos).unwrap();
                let (line_start, line_end) = file.line_bounds(line_index);

                // Update the cache entry in place
                *line = line_index + 1;
                *start = line_start;
                *end = line_end;

                return (file.clone(), line_index + 1, pos - line_start);
            }
        }

        // No cache hit ...
        let file_index = self.codemap.lookup_filemap_idx(pos);
        let file = self.codemap.files.borrow()[file_index].clone();
        let line_index = file.lookup_line(pos).unwrap();
        let (line_start, line_end) = file.line_bounds(line_index);

        // Just overwrite some cache entry. If we got this far, all of them
        // point to the wrong file.
        self.line_cache[self.eviction_index] = (line_index + 1,
                                                line_start,
                                                line_end,
                                                file.clone());
        self.eviction_index = (self.eviction_index + 1) % self.line_cache.len();

        return (file, line_index + 1, pos - line_start);
    }
}

impl<'a, 'hash, 'tcx> StrictVersionHashVisitor<'a, 'hash, 'tcx> {
    pub fn new(st: &'a mut SipHasher,
               tcx: TyCtxt<'hash, 'tcx, 'tcx>,
               def_path_hashes: &'a mut DefPathHashes<'hash, 'tcx>,
               hash_spans: bool)
               -> Self {
        StrictVersionHashVisitor {
            st: st,
            tcx: tcx,
            def_path_hashes: def_path_hashes,
            hash_spans: hash_spans,
            codemap: CachingCodemapView::new(tcx),
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
    fn hash_span(&mut self, span: Span) {
        debug_assert!(self.hash_spans);
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

        let (file1, line1, col1) = self.codemap.byte_pos_to_line_and_col(span.lo);
        let (file2, line2, col2) = self.codemap.byte_pos_to_line_and_col(span_hi);

        let expansion_kind = match span.expn_id {
            NO_EXPANSION => SawSpanExpnKind::NoExpansion,
            COMMAND_LINE_EXPN => SawSpanExpnKind::CommandLine,
            _ => SawSpanExpnKind::SomeExpansion,
        };

        SawSpan(&file1.name[..], line1, col1,
                &file2.name[..], line2, col2,
                expansion_kind).hash(self.st);

        if expansion_kind == SawSpanExpnKind::SomeExpansion {
            self.hash_span(self.codemap.codemap.source_callsite(span));
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
    SawIdent(token::InternedString),
    SawStructDef(token::InternedString),

    SawLifetime,
    SawLifetimeDef(usize),

    SawMod,
    SawForeignItem,
    SawItem,
    SawTy,
    SawGenerics,
    SawTraitItem,
    SawImplItem,
    SawStructField,
    SawVariant,
    SawPath(bool),
    SawPathSegment,
    SawPathParameters,
    SawPathListItem,
    SawBlock,
    SawPat,
    SawLocal,
    SawArm,
    SawExpr(SawExprComponent<'a>),
    SawStmt,
    SawVis,
    SawWherePredicate,
    SawTyParamBound,
    SawPolyTraitRef,
    SawAssocTypeBinding,
    SawAttribute(ast::AttrStyle),
    SawMacroDef,
    SawSpan(&'a str, usize, BytePos, &'a str, usize, BytePos, SawSpanExpnKind),
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
enum SawExprComponent<'a> {

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
    SawExprClosure(CaptureClause),
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
        ExprClosure(cc, _, _, _) => SawExprClosure(cc),
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

#[derive(Clone, Copy, Hash, Eq, PartialEq)]
enum SawSpanExpnKind {
    NoExpansion,
    CommandLine,
    SomeExpansion,
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
        if $visitor.hash_spans {
            $visitor.hash_span($span);
        }
    })
}

impl<'a, 'hash, 'tcx> visit::Visitor<'tcx> for StrictVersionHashVisitor<'a, 'hash, 'tcx> {
    fn visit_nested_item(&mut self, _: ItemId) {
        // Each item is hashed independently; ignore nested items.
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
        SawVariant.hash(self.st);
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
        SawExpr(saw_expr(&ex.node)).hash(self.st);
        // No need to explicitly hash the discriminant here, since we are
        // implicitly hashing the discriminant of SawExprComponent.
        hash_span!(self, ex.span);
        hash_attrs!(self, &ex.attrs);
        visit::walk_expr(self, ex)
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

        SawForeignItem.hash(self.st);
        hash_span!(self, i.span);
        hash_attrs!(self, &i.attrs);
        visit::walk_foreign_item(self, i)
    }

    fn visit_item(&mut self, i: &'tcx Item) {
        debug!("visit_item: {:?} st={:?}", i, self.st);

        SawItem.hash(self.st);
        // Hash the value of the discriminant of the Item variant.
        self.hash_discriminant(&i.node);
        hash_span!(self, i.span);
        hash_attrs!(self, &i.attrs);
        visit::walk_item(self, i)
    }

    fn visit_mod(&mut self, m: &'tcx Mod, _s: Span, n: NodeId) {
        debug!("visit_mod: st={:?}", self.st);
        SawMod.hash(self.st); visit::walk_mod(self, m, n)
    }

    fn visit_ty(&mut self, t: &'tcx Ty) {
        debug!("visit_ty: st={:?}", self.st);
        SawTy.hash(self.st);
        hash_span!(self, t.span);
        visit::walk_ty(self, t)
    }

    fn visit_generics(&mut self, g: &'tcx Generics) {
        debug!("visit_generics: st={:?}", self.st);
        SawGenerics.hash(self.st);
        visit::walk_generics(self, g)
    }

    fn visit_trait_item(&mut self, ti: &'tcx TraitItem) {
        debug!("visit_trait_item: st={:?}", self.st);
        SawTraitItem.hash(self.st);
        self.hash_discriminant(&ti.node);
        hash_span!(self, ti.span);
        hash_attrs!(self, &ti.attrs);
        visit::walk_trait_item(self, ti)
    }

    fn visit_impl_item(&mut self, ii: &'tcx ImplItem) {
        debug!("visit_impl_item: st={:?}", self.st);
        SawImplItem.hash(self.st);
        self.hash_discriminant(&ii.node);
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

    fn visit_path(&mut self, path: &'tcx Path, _: ast::NodeId) {
        debug!("visit_path: st={:?}", self.st);
        SawPath(path.global).hash(self.st);
        hash_span!(self, path.span);
        visit::walk_path(self, path)
    }

    fn visit_block(&mut self, b: &'tcx Block) {
        debug!("visit_block: st={:?}", self.st);
        SawBlock.hash(self.st);
        hash_span!(self, b.span);
        visit::walk_block(self, b)
    }

    fn visit_pat(&mut self, p: &'tcx Pat) {
        debug!("visit_pat: st={:?}", self.st);
        SawPat.hash(self.st);
        self.hash_discriminant(&p.node);
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

    fn visit_path_list_item(&mut self, prefix: &'tcx Path, item: &'tcx PathListItem) {
        debug!("visit_path_list_item: st={:?}", self.st);
        SawPathListItem.hash(self.st);
        self.hash_discriminant(&item.node);
        hash_span!(self, item.span);
        visit::walk_path_list_item(self, prefix, item)
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

    fn visit_attribute(&mut self, _: &Attribute) {
        // We explicitly do not use this method, since doing that would
        // implicitly impose an order on the attributes being hashed, while we
        // explicitly don't want their order to matter
    }

    fn visit_macro_def(&mut self, macro_def: &'tcx MacroDef) {
        debug!("visit_macro_def: st={:?}", self.st);
        if macro_def.export {
            SawMacroDef.hash(self.st);
            hash_attrs!(self, &macro_def.attrs);
            visit::walk_macro_def(self, macro_def)
            // FIXME(mw): We should hash the body of the macro too but we don't
            //            have a stable way of doing so yet.
        }
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

    fn hash_meta_item(&mut self, meta_item: &ast::MetaItem) {
        debug!("hash_meta_item: st={:?}", self.st);

        // ignoring span information, it doesn't matter here
        self.hash_discriminant(&meta_item.node);
        match meta_item.node {
            ast::MetaItemKind::Word(ref s) => {
                s.len().hash(self.st);
                s.hash(self.st);
            }
            ast::MetaItemKind::NameValue(ref s, ref lit) => {
                s.len().hash(self.st);
                s.hash(self.st);
                lit.node.hash(self.st);
            }
            ast::MetaItemKind::List(ref s, ref items) => {
                s.len().hash(self.st);
                s.hash(self.st);
                // Sort subitems so the hash does not depend on their order
                let indices = self.indices_sorted_by(&items, |p| {
                    meta_item_sort_key(&*p)
                });
                items.len().hash(self.st);
                for (index, &item_index) in indices.iter().enumerate() {
                    index.hash(self.st);
                    self.hash_meta_item(&items[item_index]);
                }
            }
        }
    }

    pub fn hash_attributes(&mut self, attributes: &[Attribute]) {
        debug!("hash_attributes: st={:?}", self.st);
        let indices = self.indices_sorted_by(attributes, |attr| {
            meta_item_sort_key(&attr.node.value)
        });

        for i in indices {
            let attr = &attributes[i].node;

            if !attr.is_sugared_doc &&
               !IGNORED_ATTRIBUTES.contains(&&*meta_item_sort_key(&attr.value)) {
                SawAttribute(attr.style).hash(self.st);
                self.hash_meta_item(&*attr.value);
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
}

fn meta_item_sort_key(item: &ast::MetaItem) -> token::InternedString {
    match item.node {
        ast::MetaItemKind::Word(ref s) |
        ast::MetaItemKind::NameValue(ref s, _) |
        ast::MetaItemKind::List(ref s, _) => s.clone()
    }
}
