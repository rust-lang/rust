use crate::traits::*;
use rustc::hir::def_id::CrateNum;
use rustc::mir;
use rustc::session::config::DebugInfo;
use rustc::ty::layout::{LayoutOf, Size};
use rustc::ty::TyCtxt;
use rustc_index::vec::IndexVec;

use rustc_span::symbol::kw;
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
        // FIXME(eddyb) maybe name the return place as `_0` or `return`?
        if local == mir::RETURN_PLACE {
            return;
        }

        let vars = match &self.per_local_var_debug_info {
            Some(per_local) => &per_local[local],
            None => return,
        };
        let whole_local_var = vars.iter().copied().find(|var| var.place.projection.is_empty());
        let has_proj = || vars.iter().any(|var| !var.place.projection.is_empty());

        let (fallback_var, kind) = if self.mir.local_kind(local) == mir::LocalKind::Arg {
            let arg_index = local.index() - 1;

            // Add debuginfo even to unnamed arguments.
            // FIXME(eddyb) is this really needed?
            let var = if arg_index == 0 && has_proj() {
                // Hide closure environments from debuginfo.
                // FIXME(eddyb) shouldn't `ArgumentVariable` indices
                // be offset to account for the hidden environment?
                None
            } else {
                Some(mir::VarDebugInfo {
                    name: kw::Invalid,
                    source_info: self.mir.local_decls[local].source_info,
                    place: local.into(),
                })
            };
            (var, VariableKind::ArgumentVariable(arg_index + 1))
        } else {
            (None, VariableKind::LocalVariable)
        };

        let local_ref = &self.locals[local];

        if !bx.sess().fewer_names() {
            let name = match whole_local_var.or(fallback_var.as_ref()) {
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

        if bx.sess().opts.debuginfo != DebugInfo::Full {
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

        let vars = vars.iter().copied().chain(if whole_local_var.is_none() {
            fallback_var.as_ref()
        } else {
            None
        });

        for var in vars {
            let mut layout = base.layout;
            let mut direct_offset = Size::ZERO;
            // FIXME(eddyb) use smallvec here.
            let mut indirect_offsets = vec![];

            let kind =
                if var.place.projection.is_empty() { kind } else { VariableKind::LocalVariable };

            for elem in &var.place.projection[..] {
                match *elem {
                    mir::ProjectionElem::Deref => {
                        indirect_offsets.push(Size::ZERO);
                        layout = bx.cx().layout_of(
                            layout
                                .ty
                                .builtin_deref(true)
                                .unwrap_or_else(|| {
                                    span_bug!(var.source_info.span, "cannot deref `{}`", layout.ty,)
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
                        var.place,
                    ),
                }
            }

            let (scope, span) = self.debug_loc(var.source_info);
            if let Some(scope) = scope {
                bx.declare_local(
                    debug_context,
                    var.name,
                    layout.ty,
                    scope,
                    base.llval,
                    direct_offset,
                    &indirect_offsets,
                    kind,
                    span,
                );
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
}

/// Partition all `VarDebuginfo` in `body`, by their base `Local`.
pub fn per_local_var_debug_info(
    tcx: TyCtxt<'tcx>,
    body: &'a mir::Body<'tcx>,
) -> Option<IndexVec<mir::Local, Vec<&'a mir::VarDebugInfo<'tcx>>>> {
    if tcx.sess.opts.debuginfo == DebugInfo::Full || !tcx.sess.fewer_names() {
        let mut per_local = IndexVec::from_elem(vec![], &body.local_decls);
        for var in &body.var_debug_info {
            if let mir::PlaceBase::Local(local) = var.place.base {
                per_local[local].push(var);
            }
        }
        Some(per_local)
    } else {
        None
    }
}
