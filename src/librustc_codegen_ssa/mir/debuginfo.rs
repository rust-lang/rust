use crate::traits::*;
use rustc::mir;
use rustc::session::config::DebugInfo;
use rustc::ty;
use rustc::ty::layout::{LayoutOf, Size};
use rustc_hir::def_id::CrateNum;
use rustc_index::vec::IndexVec;

use rustc_span::symbol::{kw, Symbol};
use rustc_span::{BytePos, Span};

use super::OperandValue;
use super::{FunctionCx, LocalRef};

pub struct FunctionDebugContext<D> {
    pub scopes: IndexVec<mir::SourceScope, DebugScope<D>>,
    pub source_locations_enabled: bool,
    pub defining_crate: CrateNum,
}

#[derive(Copy, Clone)]
pub enum VariableKind {
    ArgumentVariable(usize /*index*/),
    LocalVariable,
}

/// Like `mir::VarDebugInfo`, but within a `mir::Local`.
#[derive(Copy, Clone)]
pub struct PerLocalVarDebugInfo<'tcx, D> {
    pub name: Symbol,
    pub source_info: mir::SourceInfo,

    /// `DIVariable` returned by `create_dbg_var`.
    pub dbg_var: Option<D>,

    /// `.place.projection` from `mir::VarDebugInfo`.
    pub projection: &'tcx ty::List<mir::PlaceElem<'tcx>>,
}

#[derive(Clone, Copy, Debug)]
pub struct DebugScope<D> {
    pub scope_metadata: Option<D>,
    // Start and end offsets of the file to which this DIScope belongs.
    // These are used to quickly determine whether some span refers to the same file.
    pub file_start_pos: BytePos,
    pub file_end_pos: BytePos,
}

impl<D> DebugScope<D> {
    pub fn is_valid(&self) -> bool {
        !self.scope_metadata.is_none()
    }
}

impl<'a, 'tcx, Bx: BuilderMethods<'a, 'tcx>> FunctionCx<'a, 'tcx, Bx> {
    pub fn set_debug_loc(&mut self, bx: &mut Bx, source_info: mir::SourceInfo) {
        let (scope, span) = self.debug_loc(source_info);
        if let Some(debug_context) = &mut self.debug_context {
            // FIXME(eddyb) get rid of this unwrap somehow.
            bx.set_source_location(debug_context, scope.unwrap(), span);
        }
    }

    pub fn debug_loc(&self, source_info: mir::SourceInfo) -> (Option<Bx::DIScope>, Span) {
        // Bail out if debug info emission is not enabled.
        match self.debug_context {
            None => return (None, source_info.span),
            Some(_) => {}
        }

        // In order to have a good line stepping behavior in debugger, we overwrite debug
        // locations of macro expansions with that of the outermost expansion site
        // (unless the crate is being compiled with `-Z debug-macros`).
        if !source_info.span.from_expansion() || self.cx.sess().opts.debugging_opts.debug_macros {
            let scope = self.scope_metadata_for_loc(source_info.scope, source_info.span.lo());
            (scope, source_info.span)
        } else {
            // Walk up the macro expansion chain until we reach a non-expanded span.
            // We also stop at the function body level because no line stepping can occur
            // at the level above that.
            let span = rustc_span::hygiene::walk_chain(source_info.span, self.mir.span.ctxt());
            let scope = self.scope_metadata_for_loc(source_info.scope, span.lo());
            // Use span of the outermost expansion site, while keeping the original lexical scope.
            (scope, span)
        }
    }

    // DILocations inherit source file name from the parent DIScope.  Due to macro expansions
    // it may so happen that the current span belongs to a different file than the DIScope
    // corresponding to span's containing source scope.  If so, we need to create a DIScope
    // "extension" into that file.
    fn scope_metadata_for_loc(
        &self,
        scope_id: mir::SourceScope,
        pos: BytePos,
    ) -> Option<Bx::DIScope> {
        let debug_context = self.debug_context.as_ref()?;
        let scope_metadata = debug_context.scopes[scope_id].scope_metadata;
        if pos < debug_context.scopes[scope_id].file_start_pos
            || pos >= debug_context.scopes[scope_id].file_end_pos
        {
            let sm = self.cx.sess().source_map();
            let defining_crate = debug_context.defining_crate;
            Some(self.cx.extend_scope_to_file(
                scope_metadata.unwrap(),
                &sm.lookup_char_pos(pos).file,
                defining_crate,
            ))
        } else {
            scope_metadata
        }
    }

    /// Apply debuginfo and/or name, after creating the `alloca` for a local,
    /// or initializing the local with an operand (whichever applies).
    // FIXME(eddyb) use `llvm.dbg.value` (which would work for operands),
    // not just `llvm.dbg.declare` (which requires `alloca`).
    pub fn debug_introduce_local(&self, bx: &mut Bx, local: mir::Local) {
        let full_debug_info = bx.sess().opts.debuginfo == DebugInfo::Full;

        // FIXME(eddyb) maybe name the return place as `_0` or `return`?
        if local == mir::RETURN_PLACE {
            return;
        }

        let vars = match &self.per_local_var_debug_info {
            Some(per_local) => &per_local[local],
            None => return,
        };
        let whole_local_var = vars.iter().find(|var| var.projection.is_empty()).copied();
        let has_proj = || vars.iter().any(|var| !var.projection.is_empty());

        let fallback_var = if self.mir.local_kind(local) == mir::LocalKind::Arg {
            let arg_index = local.index() - 1;

            // Add debuginfo even to unnamed arguments.
            // FIXME(eddyb) is this really needed?
            if arg_index == 0 && has_proj() {
                // Hide closure environments from debuginfo.
                // FIXME(eddyb) shouldn't `ArgumentVariable` indices
                // be offset to account for the hidden environment?
                None
            } else if whole_local_var.is_some() {
                // No need to make up anything, there is a `mir::VarDebugInfo`
                // covering the whole local.
                // FIXME(eddyb) take `whole_local_var.source_info.scope` into
                // account, just in case it doesn't use `ArgumentVariable`
                // (after #67586 gets fixed).
                None
            } else {
                let name = kw::Invalid;
                let decl = &self.mir.local_decls[local];
                let (scope, span) = if full_debug_info {
                    self.debug_loc(decl.source_info)
                } else {
                    (None, decl.source_info.span)
                };
                let dbg_var = scope.map(|scope| {
                    // FIXME(eddyb) is this `+ 1` needed at all?
                    let kind = VariableKind::ArgumentVariable(arg_index + 1);

                    self.cx.create_dbg_var(
                        self.debug_context.as_ref().unwrap(),
                        name,
                        self.monomorphize(&decl.ty),
                        scope,
                        kind,
                        span,
                    )
                });

                Some(PerLocalVarDebugInfo {
                    name,
                    source_info: decl.source_info,
                    dbg_var,
                    projection: ty::List::empty(),
                })
            }
        } else {
            None
        };

        let local_ref = &self.locals[local];

        if !bx.sess().fewer_names() {
            let name = match whole_local_var.or(fallback_var) {
                Some(var) if var.name != kw::Invalid => var.name.to_string(),
                _ => format!("{:?}", local),
            };
            match local_ref {
                LocalRef::Place(place) | LocalRef::UnsizedPlace(place) => {
                    bx.set_var_name(place.llval, &name);
                }
                LocalRef::Operand(Some(operand)) => match operand.val {
                    OperandValue::Ref(x, ..) | OperandValue::Immediate(x) => {
                        bx.set_var_name(x, &name);
                    }
                    OperandValue::Pair(a, b) => {
                        // FIXME(eddyb) these are scalar components,
                        // maybe extract the high-level fields?
                        bx.set_var_name(a, &(name.clone() + ".0"));
                        bx.set_var_name(b, &(name + ".1"));
                    }
                },
                LocalRef::Operand(None) => {}
            }
        }

        if !full_debug_info {
            return;
        }

        let debug_context = match &self.debug_context {
            Some(debug_context) => debug_context,
            None => return,
        };

        // FIXME(eddyb) add debuginfo for unsized places too.
        let base = match local_ref {
            LocalRef::Place(place) => place,
            _ => return,
        };

        let vars = vars.iter().copied().chain(fallback_var);

        for var in vars {
            let mut layout = base.layout;
            let mut direct_offset = Size::ZERO;
            // FIXME(eddyb) use smallvec here.
            let mut indirect_offsets = vec![];

            for elem in &var.projection[..] {
                match *elem {
                    mir::ProjectionElem::Deref => {
                        indirect_offsets.push(Size::ZERO);
                        layout = bx.cx().layout_of(
                            layout
                                .ty
                                .builtin_deref(true)
                                .unwrap_or_else(|| {
                                    span_bug!(var.source_info.span, "cannot deref `{}`", layout.ty)
                                })
                                .ty,
                        );
                    }
                    mir::ProjectionElem::Field(field, _) => {
                        let i = field.index();
                        let offset = indirect_offsets.last_mut().unwrap_or(&mut direct_offset);
                        *offset += layout.fields.offset(i);
                        layout = layout.field(bx.cx(), i);
                    }
                    mir::ProjectionElem::Downcast(_, variant) => {
                        layout = layout.for_variant(bx.cx(), variant);
                    }
                    _ => span_bug!(
                        var.source_info.span,
                        "unsupported var debuginfo place `{:?}`",
                        mir::Place { local, projection: var.projection },
                    ),
                }
            }

            let (scope, span) = self.debug_loc(var.source_info);
            if let Some(scope) = scope {
                if let Some(dbg_var) = var.dbg_var {
                    bx.dbg_var_addr(
                        debug_context,
                        dbg_var,
                        scope,
                        base.llval,
                        direct_offset,
                        &indirect_offsets,
                        span,
                    );
                }
            }
        }
    }

    pub fn debug_introduce_locals(&self, bx: &mut Bx) {
        if bx.sess().opts.debuginfo == DebugInfo::Full || !bx.sess().fewer_names() {
            for local in self.locals.indices() {
                self.debug_introduce_local(bx, local);
            }
        }
    }

    /// Partition all `VarDebugInfo` in `self.mir`, by their base `Local`.
    pub fn compute_per_local_var_debug_info(
        &self,
    ) -> Option<IndexVec<mir::Local, Vec<PerLocalVarDebugInfo<'tcx, Bx::DIVariable>>>> {
        let full_debug_info = self.cx.sess().opts.debuginfo == DebugInfo::Full;

        if !(full_debug_info || !self.cx.sess().fewer_names()) {
            return None;
        }

        let mut per_local = IndexVec::from_elem(vec![], &self.mir.local_decls);
        for var in &self.mir.var_debug_info {
            let (scope, span) = if full_debug_info {
                self.debug_loc(var.source_info)
            } else {
                (None, var.source_info.span)
            };
            let dbg_var = scope.map(|scope| {
                let place = var.place;
                let var_ty = self.monomorphized_place_ty(place.as_ref());
                let var_kind = if self.mir.local_kind(place.local) == mir::LocalKind::Arg
                    && place.projection.is_empty()
                    && var.source_info.scope == mir::OUTERMOST_SOURCE_SCOPE
                {
                    let arg_index = place.local.index() - 1;

                    // FIXME(eddyb) shouldn't `ArgumentVariable` indices be
                    // offset in closures to account for the hidden environment?
                    // Also, is this `+ 1` needed at all?
                    VariableKind::ArgumentVariable(arg_index + 1)
                } else {
                    VariableKind::LocalVariable
                };
                self.cx.create_dbg_var(
                    self.debug_context.as_ref().unwrap(),
                    var.name,
                    var_ty,
                    scope,
                    var_kind,
                    span,
                )
            });

            per_local[var.place.local].push(PerLocalVarDebugInfo {
                name: var.name,
                source_info: var.source_info,
                dbg_var,
                projection: var.place.projection,
            });
        }
        Some(per_local)
    }
}
