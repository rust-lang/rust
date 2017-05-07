// Copyright 2012-2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.


use build;
use hair::cx::Cx;
use hair::Pattern;
use rustc::hir;
use rustc::hir::def_id::DefId;
use rustc::middle::region::CodeExtent;
use rustc::mir::*;
use rustc::mir::transform::MirSource;
use rustc::mir::visit::MutVisitor;
use rustc::traits::Reveal;
use rustc::ty::{self, Ty, TyCtxt};
use rustc::ty::subst::Substs;
use rustc::util::nodemap::NodeMap;
use rustc_data_structures::indexed_vec::{IndexVec, Idx};
use shim;
use std::mem;
use std::u32;
use syntax::abi::Abi;
use syntax::ast;
use syntax::symbol::keywords;
use syntax_pos::Span;
use util as mir_util;

/// Construct the MIR for a given def-id.
pub fn mir_build<'a, 'tcx>(tcx: TyCtxt<'a, 'tcx, 'tcx>, def_id: DefId) -> Mir<'tcx> {
    let id = tcx.hir.as_local_node_id(def_id).unwrap();
    let unsupported = || {
        span_bug!(tcx.hir.span(id), "can't build MIR for {:?}", def_id);
    };

    // Figure out what primary body this item has.
    let body_id = match tcx.hir.get(id) {
        hir::map::NodeItem(item) => {
            match item.node {
                hir::ItemConst(_, body) |
                hir::ItemStatic(_, _, body) |
                hir::ItemFn(.., body) => body,
                _ => unsupported()
            }
        }
        hir::map::NodeTraitItem(item) => {
            match item.node {
                hir::TraitItemKind::Const(_, Some(body)) |
                hir::TraitItemKind::Method(_,
                    hir::TraitMethod::Provided(body)) => body,
                _ => unsupported()
            }
        }
        hir::map::NodeImplItem(item) => {
            match item.node {
                hir::ImplItemKind::Const(_, body) |
                hir::ImplItemKind::Method(_, body) => body,
                _ => unsupported()
            }
        }
        hir::map::NodeExpr(expr) => {
            // FIXME(eddyb) Closures should have separate
            // function definition IDs and expression IDs.
            // Type-checking should not let closures get
            // this far in a constant position.
            // Assume that everything other than closures
            // is a constant "initializer" expression.
            match expr.node {
                hir::ExprClosure(_, _, body, _) => body,
                _ => hir::BodyId { node_id: expr.id }
            }
        }
        hir::map::NodeVariant(variant) =>
            return create_constructor_shim(tcx, id, &variant.node.data),
        hir::map::NodeStructCtor(ctor) =>
            return create_constructor_shim(tcx, id, ctor),
        _ => unsupported()
    };

    let src = MirSource::from_node(tcx, id);
    tcx.infer_ctxt(body_id, Reveal::UserFacing).enter(|infcx| {
        let cx = Cx::new(&infcx, src);
        let mut mir = if cx.tables().tainted_by_errors {
            build::construct_error(cx, body_id)
        } else if let MirSource::Fn(id) = src {
            // fetch the fully liberated fn signature (that is, all bound
            // types/lifetimes replaced)
            let fn_sig = cx.tables().liberated_fn_sigs[&id].clone();

            let ty = tcx.type_of(tcx.hir.local_def_id(id));
            let mut abi = fn_sig.abi;
            let implicit_argument = if let ty::TyClosure(..) = ty.sty {
                // HACK(eddyb) Avoid having RustCall on closures,
                // as it adds unnecessary (and wrong) auto-tupling.
                abi = Abi::Rust;
                Some((closure_self_ty(tcx, id, body_id), None))
            } else {
                None
            };

            let body = tcx.hir.body(body_id);
            let explicit_arguments =
                body.arguments
                    .iter()
                    .enumerate()
                    .map(|(index, arg)| {
                        (fn_sig.inputs()[index], Some(&*arg.pat))
                    });

            let arguments = implicit_argument.into_iter().chain(explicit_arguments);
            build::construct_fn(cx, id, arguments, abi, fn_sig.output(), body)
        } else {
            build::construct_const(cx, body_id)
        };

        // Convert the Mir to global types.
        let mut globalizer = GlobalizeMir {
            tcx: tcx,
            span: mir.span
        };
        globalizer.visit_mir(&mut mir);
        let mir = unsafe {
            mem::transmute::<Mir, Mir<'tcx>>(mir)
        };

        mir_util::dump_mir(tcx, None, "mir_map", &0, src, &mir);

        mir
    })
}

/// A pass to lift all the types and substitutions in a Mir
/// to the global tcx. Sadly, we don't have a "folder" that
/// can change 'tcx so we have to transmute afterwards.
struct GlobalizeMir<'a, 'gcx: 'a> {
    tcx: TyCtxt<'a, 'gcx, 'gcx>,
    span: Span
}

impl<'a, 'gcx: 'tcx, 'tcx> MutVisitor<'tcx> for GlobalizeMir<'a, 'gcx> {
    fn visit_ty(&mut self, ty: &mut Ty<'tcx>) {
        if let Some(lifted) = self.tcx.lift(ty) {
            *ty = lifted;
        } else {
            span_bug!(self.span,
                      "found type `{:?}` with inference types/regions in MIR",
                      ty);
        }
    }

    fn visit_substs(&mut self, substs: &mut &'tcx Substs<'tcx>) {
        if let Some(lifted) = self.tcx.lift(substs) {
            *substs = lifted;
        } else {
            span_bug!(self.span,
                      "found substs `{:?}` with inference types/regions in MIR",
                      substs);
        }
    }
}

fn create_constructor_shim<'a, 'tcx>(tcx: TyCtxt<'a, 'tcx, 'tcx>,
                                     ctor_id: ast::NodeId,
                                     v: &'tcx hir::VariantData)
                                     -> Mir<'tcx>
{
    let span = tcx.hir.span(ctor_id);
    if let hir::VariantData::Tuple(ref fields, ctor_id) = *v {
        let pe = tcx.parameter_environment(tcx.hir.local_def_id(ctor_id));
        tcx.infer_ctxt(pe, Reveal::UserFacing).enter(|infcx| {
            let (mut mir, src) =
                shim::build_adt_ctor(&infcx, ctor_id, fields, span);

            // Convert the Mir to global types.
            let tcx = infcx.tcx.global_tcx();
            let mut globalizer = GlobalizeMir {
                tcx: tcx,
                span: mir.span
            };
            globalizer.visit_mir(&mut mir);
            let mir = unsafe {
                mem::transmute::<Mir, Mir<'tcx>>(mir)
            };

            mir_util::dump_mir(tcx, None, "mir_map", &0, src, &mir);

            mir
        })
    } else {
        span_bug!(span, "attempting to create MIR for non-tuple variant {:?}", v);
    }
}

///////////////////////////////////////////////////////////////////////////
// BuildMir -- walks a crate, looking for fn items and methods to build MIR from

fn closure_self_ty<'a, 'tcx>(tcx: TyCtxt<'a, 'tcx, 'tcx>,
                             closure_expr_id: ast::NodeId,
                             body_id: hir::BodyId)
                             -> Ty<'tcx> {
    let closure_ty = tcx.body_tables(body_id).node_id_to_type(closure_expr_id);

    let region = ty::ReFree(ty::FreeRegion {
        scope: Some(tcx.item_extent(body_id.node_id)),
        bound_region: ty::BoundRegion::BrEnv,
    });
    let region = tcx.mk_region(region);

    match tcx.closure_kind(tcx.hir.local_def_id(closure_expr_id)) {
        ty::ClosureKind::Fn =>
            tcx.mk_ref(region,
                       ty::TypeAndMut { ty: closure_ty,
                                        mutbl: hir::MutImmutable }),
        ty::ClosureKind::FnMut =>
            tcx.mk_ref(region,
                       ty::TypeAndMut { ty: closure_ty,
                                        mutbl: hir::MutMutable }),
        ty::ClosureKind::FnOnce =>
            closure_ty
    }
}

struct Builder<'a, 'gcx: 'a+'tcx, 'tcx: 'a> {
    hir: Cx<'a, 'gcx, 'tcx>,
    cfg: CFG<'tcx>,

    fn_span: Span,
    arg_count: usize,

    /// the current set of scopes, updated as we traverse;
    /// see the `scope` module for more details
    scopes: Vec<scope::Scope<'tcx>>,

    /// the current set of breakables; see the `scope` module for more
    /// details
    breakable_scopes: Vec<scope::BreakableScope<'tcx>>,

    /// the vector of all scopes that we have created thus far;
    /// we track this for debuginfo later
    visibility_scopes: IndexVec<VisibilityScope, VisibilityScopeData>,
    visibility_scope: VisibilityScope,

    /// Maps node ids of variable bindings to the `Local`s created for them.
    var_indices: NodeMap<Local>,
    local_decls: IndexVec<Local, LocalDecl<'tcx>>,
    unit_temp: Option<Lvalue<'tcx>>,

    /// cached block with the RESUME terminator; this is created
    /// when first set of cleanups are built.
    cached_resume_block: Option<BasicBlock>,
    /// cached block with the RETURN terminator
    cached_return_block: Option<BasicBlock>,
}

struct CFG<'tcx> {
    basic_blocks: IndexVec<BasicBlock, BasicBlockData<'tcx>>,
}

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct ScopeId(u32);

impl Idx for ScopeId {
    fn new(index: usize) -> ScopeId {
        assert!(index < (u32::MAX as usize));
        ScopeId(index as u32)
    }

    fn index(self) -> usize {
        self.0 as usize
    }
}

///////////////////////////////////////////////////////////////////////////
/// The `BlockAnd` "monad" packages up the new basic block along with a
/// produced value (sometimes just unit, of course). The `unpack!`
/// macro (and methods below) makes working with `BlockAnd` much more
/// convenient.

#[must_use] // if you don't use one of these results, you're leaving a dangling edge
struct BlockAnd<T>(BasicBlock, T);

trait BlockAndExtension {
    fn and<T>(self, v: T) -> BlockAnd<T>;
    fn unit(self) -> BlockAnd<()>;
}

impl BlockAndExtension for BasicBlock {
    fn and<T>(self, v: T) -> BlockAnd<T> {
        BlockAnd(self, v)
    }

    fn unit(self) -> BlockAnd<()> {
        BlockAnd(self, ())
    }
}

/// Update a block pointer and return the value.
/// Use it like `let x = unpack!(block = self.foo(block, foo))`.
macro_rules! unpack {
    ($x:ident = $c:expr) => {
        {
            let BlockAnd(b, v) = $c;
            $x = b;
            v
        }
    };

    ($c:expr) => {
        {
            let BlockAnd(b, ()) = $c;
            b
        }
    };
}

///////////////////////////////////////////////////////////////////////////
/// the main entry point for building MIR for a function

fn construct_fn<'a, 'gcx, 'tcx, A>(hir: Cx<'a, 'gcx, 'tcx>,
                                   fn_id: ast::NodeId,
                                   arguments: A,
                                   abi: Abi,
                                   return_ty: Ty<'gcx>,
                                   body: &'gcx hir::Body)
                                   -> Mir<'tcx>
    where A: Iterator<Item=(Ty<'gcx>, Option<&'gcx hir::Pat>)>
{
    let arguments: Vec<_> = arguments.collect();

    let tcx = hir.tcx();
    let span = tcx.hir.span(fn_id);
    let mut builder = Builder::new(hir.clone(), span, arguments.len(), return_ty);

    let call_site_extent = tcx.call_site_extent(fn_id);
    let arg_extent = tcx.parameter_extent(fn_id);
    let mut block = START_BLOCK;
    unpack!(block = builder.in_scope(call_site_extent, block, |builder| {
        unpack!(block = builder.in_scope(arg_extent, block, |builder| {
            builder.args_and_body(block, &arguments, arg_extent, &body.value)
        }));
        // Attribute epilogue to function's closing brace
        let fn_end = Span { lo: span.hi, ..span };
        let source_info = builder.source_info(fn_end);
        let return_block = builder.return_block();
        builder.cfg.terminate(block, source_info,
                              TerminatorKind::Goto { target: return_block });
        builder.cfg.terminate(return_block, source_info,
                              TerminatorKind::Return);
        return_block.unit()
    }));
    assert_eq!(block, builder.return_block());

    let mut spread_arg = None;
    if abi == Abi::RustCall {
        // RustCall pseudo-ABI untuples the last argument.
        spread_arg = Some(Local::new(arguments.len()));
    }

    // Gather the upvars of a closure, if any.
    let upvar_decls: Vec<_> = tcx.with_freevars(fn_id, |freevars| {
        freevars.iter().map(|fv| {
            let var_id = tcx.hir.as_local_node_id(fv.def.def_id()).unwrap();
            let by_ref = hir.tables().upvar_capture(ty::UpvarId {
                var_id: var_id,
                closure_expr_id: fn_id
            }).map_or(false, |capture| match capture {
                ty::UpvarCapture::ByValue => false,
                ty::UpvarCapture::ByRef(..) => true
            });
            let mut decl = UpvarDecl {
                debug_name: keywords::Invalid.name(),
                by_ref: by_ref
            };
            if let Some(hir::map::NodeLocal(pat)) = tcx.hir.find(var_id) {
                if let hir::PatKind::Binding(_, _, ref ident, _) = pat.node {
                    decl.debug_name = ident.node;
                }
            }
            decl
        }).collect()
    });

    let mut mir = builder.finish(upvar_decls, return_ty);
    mir.spread_arg = spread_arg;
    mir
}

pub fn construct_const<'a, 'gcx, 'tcx>(hir: Cx<'a, 'gcx, 'tcx>,
                                       body_id: hir::BodyId)
                                       -> Mir<'tcx> {
    let tcx = hir.tcx();
    let ast_expr = &tcx.hir.body(body_id).value;
    let ty = hir.tables().expr_ty_adjusted(ast_expr);
    let owner_id = tcx.hir.body_owner(body_id);
    let span = tcx.hir.span(owner_id);
    let mut builder = Builder::new(hir.clone(), span, 0, ty);

    let extent = hir.region_maps.temporary_scope(tcx, ast_expr.id)
                                .unwrap_or(tcx.item_extent(owner_id));
    let mut block = START_BLOCK;
    let _ = builder.in_scope(extent, block, |builder| {
        let expr = builder.hir.mirror(ast_expr);
        unpack!(block = builder.into(&Lvalue::Local(RETURN_POINTER), block, expr));

        let source_info = builder.source_info(span);
        let return_block = builder.return_block();
        builder.cfg.terminate(block, source_info,
                              TerminatorKind::Goto { target: return_block });
        builder.cfg.terminate(return_block, source_info,
                              TerminatorKind::Return);

        return_block.unit()
    });

    builder.finish(vec![], ty)
}

pub fn construct_error<'a, 'gcx, 'tcx>(hir: Cx<'a, 'gcx, 'tcx>,
                                       body_id: hir::BodyId)
                                       -> Mir<'tcx> {
    let span = hir.tcx().hir.span(hir.tcx().hir.body_owner(body_id));
    let ty = hir.tcx().types.err;
    let mut builder = Builder::new(hir, span, 0, ty);
    let source_info = builder.source_info(span);
    builder.cfg.terminate(START_BLOCK, source_info, TerminatorKind::Unreachable);
    builder.finish(vec![], ty)
}

impl<'a, 'gcx, 'tcx> Builder<'a, 'gcx, 'tcx> {
    fn new(hir: Cx<'a, 'gcx, 'tcx>,
           span: Span,
           arg_count: usize,
           return_ty: Ty<'tcx>)
           -> Builder<'a, 'gcx, 'tcx> {
        let mut builder = Builder {
            hir: hir,
            cfg: CFG { basic_blocks: IndexVec::new() },
            fn_span: span,
            arg_count: arg_count,
            scopes: vec![],
            visibility_scopes: IndexVec::new(),
            visibility_scope: ARGUMENT_VISIBILITY_SCOPE,
            breakable_scopes: vec![],
            local_decls: IndexVec::from_elem_n(LocalDecl::new_return_pointer(return_ty,
                                                                             span), 1),
            var_indices: NodeMap(),
            unit_temp: None,
            cached_resume_block: None,
            cached_return_block: None
        };

        assert_eq!(builder.cfg.start_new_block(), START_BLOCK);
        assert_eq!(builder.new_visibility_scope(span), ARGUMENT_VISIBILITY_SCOPE);
        builder.visibility_scopes[ARGUMENT_VISIBILITY_SCOPE].parent_scope = None;

        builder
    }

    fn finish(self,
              upvar_decls: Vec<UpvarDecl>,
              return_ty: Ty<'tcx>)
              -> Mir<'tcx> {
        for (index, block) in self.cfg.basic_blocks.iter().enumerate() {
            if block.terminator.is_none() {
                span_bug!(self.fn_span, "no terminator on block {:?}", index);
            }
        }

        Mir::new(self.cfg.basic_blocks,
                 self.visibility_scopes,
                 IndexVec::new(),
                 return_ty,
                 self.local_decls,
                 self.arg_count,
                 upvar_decls,
                 self.fn_span
        )
    }

    fn args_and_body(&mut self,
                     mut block: BasicBlock,
                     arguments: &[(Ty<'gcx>, Option<&'gcx hir::Pat>)],
                     argument_extent: CodeExtent<'tcx>,
                     ast_body: &'gcx hir::Expr)
                     -> BlockAnd<()>
    {
        // Allocate locals for the function arguments
        for &(ty, pattern) in arguments.iter() {
            // If this is a simple binding pattern, give the local a nice name for debuginfo.
            let mut name = None;
            if let Some(pat) = pattern {
                if let hir::PatKind::Binding(_, _, ref ident, _) = pat.node {
                    name = Some(ident.node);
                }
            }

            self.local_decls.push(LocalDecl {
                mutability: Mutability::Not,
                ty: ty,
                source_info: SourceInfo {
                    scope: ARGUMENT_VISIBILITY_SCOPE,
                    span: pattern.map_or(self.fn_span, |pat| pat.span)
                },
                name: name,
                is_user_variable: false,
            });
        }

        let mut scope = None;
        // Bind the argument patterns
        for (index, &(ty, pattern)) in arguments.iter().enumerate() {
            // Function arguments always get the first Local indices after the return pointer
            let lvalue = Lvalue::Local(Local::new(index + 1));

            if let Some(pattern) = pattern {
                let pattern = Pattern::from_hir(self.hir.tcx(), self.hir.tables(), pattern);
                scope = self.declare_bindings(scope, ast_body.span, &pattern);
                unpack!(block = self.lvalue_into_pattern(block, pattern, &lvalue));
            }

            // Make sure we drop (parts of) the argument even when not matched on.
            self.schedule_drop(pattern.as_ref().map_or(ast_body.span, |pat| pat.span),
                               argument_extent, &lvalue, ty);

        }

        // Enter the argument pattern bindings visibility scope, if it exists.
        if let Some(visibility_scope) = scope {
            self.visibility_scope = visibility_scope;
        }

        let body = self.hir.mirror(ast_body);
        self.into(&Lvalue::Local(RETURN_POINTER), block, body)
    }

    fn get_unit_temp(&mut self) -> Lvalue<'tcx> {
        match self.unit_temp {
            Some(ref tmp) => tmp.clone(),
            None => {
                let ty = self.hir.unit_ty();
                let fn_span = self.fn_span;
                let tmp = self.temp(ty, fn_span);
                self.unit_temp = Some(tmp.clone());
                tmp
            }
        }
    }

    fn return_block(&mut self) -> BasicBlock {
        match self.cached_return_block {
            Some(rb) => rb,
            None => {
                let rb = self.cfg.start_new_block();
                self.cached_return_block = Some(rb);
                rb
            }
        }
    }
}

///////////////////////////////////////////////////////////////////////////
// Builder methods are broken up into modules, depending on what kind
// of thing is being translated. Note that they use the `unpack` macro
// above extensively.

mod block;
mod cfg;
mod expr;
mod into;
mod matches;
mod misc;
mod scope;
