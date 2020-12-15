use crate::traits::*;
use rustc_index::vec::IndexVec;
use rustc_middle::middle::codegen_fn_attrs::CodegenFnAttrFlags;
use rustc_middle::mir;
use rustc_middle::ty;
use rustc_session::config::DebugInfo;
use rustc_span::symbol::{kw, Symbol};
use rustc_span::{BytePos, Span};
use rustc_target::abi::{LayoutOf, Size};

use super::operand::{OperandRef, OperandValue};
use super::place::PlaceRef;
use super::{FunctionCx, LocalRef};

pub struct FunctionDebugContext<S, L> {
    pub scopes: IndexVec<mir::SourceScope, DebugScope<S, L>>,
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
pub struct DebugScope<S, L> {
    // FIXME(eddyb) this should never be `None`, after initialization.
    pub dbg_scope: Option<S>,

    /// Call site location, if this scope was inlined from another function.
    pub inlined_at: Option<L>,

    // Start and end offsets of the file to which this DIScope belongs.
    // These are used to quickly determine whether some span refers to the same file.
    pub file_start_pos: BytePos,
    pub file_end_pos: BytePos,
}

impl<'tcx, S: Copy, L: Copy> DebugScope<S, L> {
    /// DILocations inherit source file name from the parent DIScope.  Due to macro expansions
    /// it may so happen that the current span belongs to a different file than the DIScope
    /// corresponding to span's containing source scope.  If so, we need to create a DIScope
    /// "extension" into that file.
    pub fn adjust_dbg_scope_for_span<Cx: CodegenMethods<'tcx, DIScope = S, DILocation = L>>(
        &self,
        cx: &Cx,
        span: Span,
    ) -> S {
        // FIXME(eddyb) this should never be `None`.
        let dbg_scope = self
            .dbg_scope
            .unwrap_or_else(|| bug!("`dbg_scope` is only `None` during initialization"));

        let pos = span.lo();
        if pos < self.file_start_pos || pos >= self.file_end_pos {
            let sm = cx.sess().source_map();
            cx.extend_scope_to_file(dbg_scope, &sm.lookup_char_pos(pos).file)
        } else {
            dbg_scope
        }
    }
}

impl<'a, 'tcx, Bx: BuilderMethods<'a, 'tcx>> FunctionCx<'a, 'tcx, Bx> {
    pub fn set_debug_loc(&self, bx: &mut Bx, source_info: mir::SourceInfo) {
        bx.set_span(source_info.span);
        if let Some(dbg_loc) = self.dbg_loc(source_info) {
            bx.set_dbg_loc(dbg_loc);
        }
    }

    fn dbg_loc(&self, source_info: mir::SourceInfo) -> Option<Bx::DILocation> {
        let (dbg_scope, inlined_at, span) = self.adjusted_span_and_dbg_scope(source_info)?;
        Some(self.cx.dbg_loc(dbg_scope, inlined_at, span))
    }

    fn adjusted_span_and_dbg_scope(
        &self,
        source_info: mir::SourceInfo,
    ) -> Option<(Bx::DIScope, Option<Bx::DILocation>, Span)> {
        let span = self.adjust_span_for_debugging(source_info.span);
        let scope = &self.debug_context.as_ref()?.scopes[source_info.scope];
        Some((scope.adjust_dbg_scope_for_span(self.cx, span), scope.inlined_at, span))
    }

    /// In order to have a good line stepping behavior in debugger, we overwrite debug
    /// locations of macro expansions with that of the outermost expansion site
    /// (unless the crate is being compiled with `-Z debug-macros`).
    fn adjust_span_for_debugging(&self, mut span: Span) -> Span {
        // Bail out if debug info emission is not enabled.
        if self.debug_context.is_none() {
            return span;
        }

        if span.from_expansion() && !self.cx.sess().opts.debugging_opts.debug_macros {
            // Walk up the macro expansion chain until we reach a non-expanded span.
            // We also stop at the function body level because no line stepping can occur
            // at the level above that.
            // Use span of the outermost expansion site, while keeping the original lexical scope.
            span = rustc_span::hygiene::walk_chain(span, self.mir.span.ctxt());
        }

        span
    }

    fn spill_operand_to_stack(
        operand: &OperandRef<'tcx, Bx::Value>,
        name: Option<String>,
        bx: &mut Bx,
    ) -> PlaceRef<'tcx, Bx::Value> {
        // "Spill" the value onto the stack, for debuginfo,
        // without forcing non-debuginfo uses of the local
        // to also load from the stack every single time.
        // FIXME(#68817) use `llvm.dbg.value` instead,
        // at least for the cases which LLVM handles correctly.
        let spill_slot = PlaceRef::alloca(bx, operand.layout);
        if let Some(name) = name {
            bx.set_var_name(spill_slot.llval, &(name + ".dbg.spill"));
        }
        operand.val.store(bx, spill_slot);
        spill_slot
    }

    /// Apply debuginfo and/or name, after creating the `alloca` for a local,
    /// or initializing the local with an operand (whichever applies).
    pub fn debug_introduce_local(&self, bx: &mut Bx, local: mir::Local) {
        let full_debug_info = bx.sess().opts.debuginfo == DebugInfo::Full;

        // FIXME(eddyb) maybe name the return place as `_0` or `return`?
        if local == mir::RETURN_PLACE && !self.mir.local_decls[mir::RETURN_PLACE].is_user_variable()
        {
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
                let dbg_var = if full_debug_info {
                    self.adjusted_span_and_dbg_scope(decl.source_info).map(
                        |(dbg_scope, _, span)| {
                            // FIXME(eddyb) is this `+ 1` needed at all?
                            let kind = VariableKind::ArgumentVariable(arg_index + 1);

                            let arg_ty = self.monomorphize(decl.ty);

                            self.cx.create_dbg_var(name, arg_ty, dbg_scope, kind, span)
                        },
                    )
                } else {
                    None
                };

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

        let name = if bx.sess().fewer_names() {
            None
        } else {
            Some(match whole_local_var.or(fallback_var) {
                Some(var) if var.name != kw::Invalid => var.name.to_string(),
                _ => format!("{:?}", local),
            })
        };

        if let Some(name) = &name {
            match local_ref {
                LocalRef::Place(place) | LocalRef::UnsizedPlace(place) => {
                    bx.set_var_name(place.llval, name);
                }
                LocalRef::Operand(Some(operand)) => match operand.val {
                    OperandValue::Ref(x, ..) | OperandValue::Immediate(x) => {
                        bx.set_var_name(x, name);
                    }
                    OperandValue::Pair(a, b) => {
                        // FIXME(eddyb) these are scalar components,
                        // maybe extract the high-level fields?
                        bx.set_var_name(a, &(name.clone() + ".0"));
                        bx.set_var_name(b, &(name.clone() + ".1"));
                    }
                },
                LocalRef::Operand(None) => {}
            }
        }

        if !full_debug_info || vars.is_empty() && fallback_var.is_none() {
            return;
        }

        let base = match local_ref {
            LocalRef::Operand(None) => return,

            LocalRef::Operand(Some(operand)) => {
                // Don't spill operands onto the stack in naked functions.
                // See: https://github.com/rust-lang/rust/issues/42779
                let attrs = bx.tcx().codegen_fn_attrs(self.instance.def_id());
                if attrs.flags.contains(CodegenFnAttrFlags::NAKED) {
                    return;
                }

                Self::spill_operand_to_stack(operand, name, bx)
            }

            LocalRef::Place(place) => *place,

            // FIXME(eddyb) add debuginfo for unsized places too.
            LocalRef::UnsizedPlace(_) => return,
        };

        let vars = vars.iter().copied().chain(fallback_var);

        for var in vars {
            let dbg_var = match var.dbg_var {
                Some(dbg_var) => dbg_var,
                None => continue,
            };
            let dbg_loc = match self.dbg_loc(var.source_info) {
                Some(dbg_loc) => dbg_loc,
                None => continue,
            };

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

            bx.dbg_var_addr(dbg_var, dbg_loc, base.llval, direct_offset, &indirect_offsets);
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
        bx: &mut Bx,
    ) -> Option<IndexVec<mir::Local, Vec<PerLocalVarDebugInfo<'tcx, Bx::DIVariable>>>> {
        let full_debug_info = self.cx.sess().opts.debuginfo == DebugInfo::Full;

        if !full_debug_info && self.cx.sess().fewer_names() {
            return None;
        }

        let mut per_local = IndexVec::from_elem(vec![], &self.mir.local_decls);
        for var in &self.mir.var_debug_info {
            let dbg_scope_and_span = if full_debug_info {
                self.adjusted_span_and_dbg_scope(var.source_info)
            } else {
                None
            };

            let dbg_var = dbg_scope_and_span.map(|(dbg_scope, _, span)| {
                let (var_ty, var_kind) = match var.value {
                    mir::VarDebugInfoContents::Place(place) => {
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
                        (var_ty, var_kind)
                    }
                    mir::VarDebugInfoContents::Const(c) => {
                        let ty = self.monomorphize(c.literal.ty);
                        (ty, VariableKind::LocalVariable)
                    }
                };

                self.cx.create_dbg_var(var.name, var_ty, dbg_scope, var_kind, span)
            });

            match var.value {
                mir::VarDebugInfoContents::Place(place) => {
                    per_local[place.local].push(PerLocalVarDebugInfo {
                        name: var.name,
                        source_info: var.source_info,
                        dbg_var,
                        projection: place.projection,
                    });
                }
                mir::VarDebugInfoContents::Const(c) => {
                    if let Some(dbg_var) = dbg_var {
                        let dbg_loc = match self.dbg_loc(var.source_info) {
                            Some(dbg_loc) => dbg_loc,
                            None => continue,
                        };

                        if let Ok(operand) = self.eval_mir_constant_to_operand(bx, &c) {
                            let base = Self::spill_operand_to_stack(
                                &operand,
                                Some(var.name.to_string()),
                                bx,
                            );

                            bx.dbg_var_addr(dbg_var, dbg_loc, base.llval, Size::ZERO, &[]);
                        }
                    }
                }
            }
        }
        Some(per_local)
    }
}
