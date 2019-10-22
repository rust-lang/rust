use rustc_index::vec::Idx;
use rustc::hir::def_id::CrateNum;
use rustc::mir;
use rustc::session::config::DebugInfo;
use rustc::ty::{self, UpvarSubsts};
use rustc::ty::layout::HasTyCtxt;
use rustc_target::abi::{Variants, VariantIdx};
use crate::traits::*;

use syntax_pos::{DUMMY_SP, BytePos, Span};
use syntax::symbol::kw;

use super::{FunctionCx, LocalRef};
use super::OperandValue;

pub enum FunctionDebugContext<D> {
    RegularContext(FunctionDebugContextData<D>),
    DebugInfoDisabled,
    FunctionWithoutDebugInfo,
}

impl<D> FunctionDebugContext<D> {
    pub fn get_ref(&self, span: Span) -> &FunctionDebugContextData<D> {
        match *self {
            FunctionDebugContext::RegularContext(ref data) => data,
            FunctionDebugContext::DebugInfoDisabled => {
                span_bug!(
                    span,
                    "debuginfo: Error trying to access FunctionDebugContext \
                     although debug info is disabled!",
                );
            }
            FunctionDebugContext::FunctionWithoutDebugInfo => {
                span_bug!(
                    span,
                    "debuginfo: Error trying to access FunctionDebugContext \
                     for function that should be ignored by debug info!",
                );
            }
        }
    }
}

/// Enables emitting source locations for the given functions.
///
/// Since we don't want source locations to be emitted for the function prelude,
/// they are disabled when beginning to codegen a new function. This functions
/// switches source location emitting on and must therefore be called before the
/// first real statement/expression of the function is codegened.
pub fn start_emitting_source_locations<D>(dbg_context: &mut FunctionDebugContext<D>) {
    match *dbg_context {
        FunctionDebugContext::RegularContext(ref mut data) => {
            data.source_locations_enabled = true;
        },
        _ => { /* safe to ignore */ }
    }
}

pub struct FunctionDebugContextData<D> {
    pub fn_metadata: D,
    pub source_locations_enabled: bool,
    pub defining_crate: CrateNum,
}

pub enum VariableAccess<'a, V> {
    // The llptr given is an alloca containing the variable's value
    DirectVariable { alloca: V },
    // The llptr given is an alloca containing the start of some pointer chain
    // leading to the variable's content.
    IndirectVariable { alloca: V, address_operations: &'a [i64] }
}

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
    pub fn set_debug_loc(
        &mut self,
        bx: &mut Bx,
        source_info: mir::SourceInfo
    ) {
        let (scope, span) = self.debug_loc(source_info);
        bx.set_source_location(&mut self.debug_context, scope, span);
    }

    pub fn debug_loc(&self, source_info: mir::SourceInfo) -> (Option<Bx::DIScope>, Span) {
        // Bail out if debug info emission is not enabled.
        match self.debug_context {
            FunctionDebugContext::DebugInfoDisabled |
            FunctionDebugContext::FunctionWithoutDebugInfo => {
                return (self.scopes[source_info.scope].scope_metadata, source_info.span);
            }
            FunctionDebugContext::RegularContext(_) =>{}
        }

        // In order to have a good line stepping behavior in debugger, we overwrite debug
        // locations of macro expansions with that of the outermost expansion site
        // (unless the crate is being compiled with `-Z debug-macros`).
        if !source_info.span.from_expansion() ||
           self.cx.sess().opts.debugging_opts.debug_macros {
            let scope = self.scope_metadata_for_loc(source_info.scope, source_info.span.lo());
            (scope, source_info.span)
        } else {
            // Walk up the macro expansion chain until we reach a non-expanded span.
            // We also stop at the function body level because no line stepping can occur
            // at the level above that.
            let span = syntax_pos::hygiene::walk_chain(source_info.span, self.mir.span.ctxt());
            let scope = self.scope_metadata_for_loc(source_info.scope, span.lo());
            // Use span of the outermost expansion site, while keeping the original lexical scope.
            (scope, span)
        }
    }

    // DILocations inherit source file name from the parent DIScope.  Due to macro expansions
    // it may so happen that the current span belongs to a different file than the DIScope
    // corresponding to span's containing source scope.  If so, we need to create a DIScope
    // "extension" into that file.
    fn scope_metadata_for_loc(&self, scope_id: mir::SourceScope, pos: BytePos)
                              -> Option<Bx::DIScope> {
        let scope_metadata = self.scopes[scope_id].scope_metadata;
        if pos < self.scopes[scope_id].file_start_pos ||
           pos >= self.scopes[scope_id].file_end_pos {
            let sm = self.cx.sess().source_map();
            let defining_crate = self.debug_context.get_ref(DUMMY_SP).defining_crate;
            Some(self.cx.extend_scope_to_file(
                scope_metadata.unwrap(),
                &sm.lookup_char_pos(pos).file,
                defining_crate
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
        let upvar_debuginfo = &self.mir.__upvar_debuginfo_codegen_only_do_not_use;

        // FIXME(eddyb) maybe name the return place as `_0` or `return`?
        if local == mir::RETURN_PLACE {
            return;
        }

        let decl = &self.mir.local_decls[local];
        let (name, kind) = if self.mir.local_kind(local) == mir::LocalKind::Arg {
            let arg_index = local.index() - 1;

            // Add debuginfo even to unnamed arguments.
            // FIXME(eddyb) is this really needed?
            let name = if arg_index == 0 && !upvar_debuginfo.is_empty() {
                // Hide closure environments from debuginfo.
                // FIXME(eddyb) shouldn't `ArgumentVariable` indices
                // be offset to account for the hidden environment?
                None
            } else {
                Some(decl.name.unwrap_or(kw::Invalid))
            };
            (name, VariableKind::ArgumentVariable(arg_index + 1))
        } else {
            (decl.name, VariableKind::LocalVariable)
        };

        let local_ref = &self.locals[local];

        if !bx.sess().fewer_names() {
            let name = match name {
                Some(name) if name != kw::Invalid => name.to_string(),
                _ => format!("{:?}", local),
            };
            match local_ref {
                LocalRef::Place(place) |
                LocalRef::UnsizedPlace(place) => {
                    bx.set_var_name(place.llval, &name);
                }
                LocalRef::Operand(Some(operand)) => match operand.val {
                    OperandValue::Ref(x, ..) |
                    OperandValue::Immediate(x) => {
                        bx.set_var_name(x, &name);
                    }
                    OperandValue::Pair(a, b) => {
                        // FIXME(eddyb) these are scalar components,
                        // maybe extract the high-level fields?
                        bx.set_var_name(a, &(name.clone() + ".0"));
                        bx.set_var_name(b, &(name + ".1"));
                    }
                }
                LocalRef::Operand(None) => {}
            }
        }

        if let Some(name) = name {
            if bx.sess().opts.debuginfo != DebugInfo::Full {
                return;
            }

            // FIXME(eddyb) add debuginfo for unsized places too.
            let place = match local_ref {
                LocalRef::Place(place) => place,
                _ => return,
            };

            let (scope, span) = self.debug_loc(mir::SourceInfo {
                span: decl.source_info.span,
                scope: decl.visibility_scope,
            });
            if let Some(scope) = scope {
                bx.declare_local(&self.debug_context, name, place.layout.ty, scope,
                    VariableAccess::DirectVariable { alloca: place.llval },
                    kind, span);
            }
        }
    }

    pub fn debug_introduce_locals(&self, bx: &mut Bx) {
        let tcx = self.cx.tcx();
        let upvar_debuginfo = &self.mir.__upvar_debuginfo_codegen_only_do_not_use;

        if bx.sess().opts.debuginfo != DebugInfo::Full {
            // HACK(eddyb) figure out a way to perhaps disentangle
            // the use of `declare_local` and `set_var_name`.
            // Or maybe just running this loop always is not that expensive?
            if !bx.sess().fewer_names() {
                for local in self.locals.indices() {
                    self.debug_introduce_local(bx, local);
                }
            }

            return;
        }

        for local in self.locals.indices() {
            self.debug_introduce_local(bx, local);
        }

        // Declare closure captures as if they were local variables.
        // FIXME(eddyb) generalize this to `name => place` mappings.
        let upvar_scope = if !upvar_debuginfo.is_empty() {
            self.scopes[mir::OUTERMOST_SOURCE_SCOPE].scope_metadata
        } else {
            None
        };
        if let Some(scope) = upvar_scope {
            let place = match self.locals[mir::Local::new(1)] {
                LocalRef::Place(place) => place,
                _ => bug!(),
            };

            let pin_did = tcx.lang_items().pin_type();
            let (closure_layout, env_ref) = match place.layout.ty.kind {
                ty::RawPtr(ty::TypeAndMut { ty, .. }) |
                ty::Ref(_, ty, _)  => (bx.layout_of(ty), true),
                ty::Adt(def, substs) if Some(def.did) == pin_did => {
                    match substs.type_at(0).kind {
                        ty::Ref(_, ty, _)  => (bx.layout_of(ty), true),
                        _ => (place.layout, false),
                    }
                }
                _ => (place.layout, false)
            };

            let (def_id, upvar_substs) = match closure_layout.ty.kind {
                ty::Closure(def_id, substs) => (def_id, UpvarSubsts::Closure(substs)),
                ty::Generator(def_id, substs, _) => (def_id, UpvarSubsts::Generator(substs)),
                _ => bug!("upvar debuginfo with non-closure arg0 type `{}`", closure_layout.ty)
            };
            let upvar_tys = upvar_substs.upvar_tys(def_id, tcx);

            let extra_locals = {
                let upvars = upvar_debuginfo
                    .iter()
                    .zip(upvar_tys)
                    .enumerate()
                    .map(|(i, (upvar, ty))| {
                        (None, i, upvar.debug_name, upvar.by_ref, ty, scope, DUMMY_SP)
                    });

                let generator_fields = self.mir.generator_layout.as_ref().map(|generator_layout| {
                    let (def_id, gen_substs) = match closure_layout.ty.kind {
                        ty::Generator(def_id, substs, _) => (def_id, substs),
                        _ => bug!("generator layout without generator substs"),
                    };
                    let state_tys = gen_substs.as_generator().state_tys(def_id, tcx);

                    generator_layout.variant_fields.iter()
                        .zip(state_tys)
                        .enumerate()
                        .flat_map(move |(variant_idx, (fields, tys))| {
                            let variant_idx = Some(VariantIdx::from(variant_idx));
                            fields.iter()
                                .zip(tys)
                                .enumerate()
                                .filter_map(move |(i, (field, ty))| {
                                    let decl = &generator_layout.
                                        __local_debuginfo_codegen_only_do_not_use[*field];
                                    if let Some(name) = decl.name {
                                        let ty = self.monomorphize(&ty);
                                        let (var_scope, var_span) = self.debug_loc(mir::SourceInfo {
                                            span: decl.source_info.span,
                                            scope: decl.visibility_scope,
                                        });
                                        let var_scope = var_scope.unwrap_or(scope);
                                        Some((variant_idx, i, name, false, ty, var_scope, var_span))
                                    } else {
                                        None
                                    }
                            })
                        })
                }).into_iter().flatten();

                upvars.chain(generator_fields)
            };

            for (variant_idx, field, name, by_ref, ty, var_scope, var_span) in extra_locals {
                let fields = match variant_idx {
                    Some(variant_idx) => {
                        match &closure_layout.variants {
                            Variants::Multiple { variants, .. } => {
                                &variants[variant_idx].fields
                            },
                            _ => bug!("variant index on univariant layout"),
                        }
                    }
                    None => &closure_layout.fields,
                };
                let byte_offset_of_var_in_env = fields.offset(field).bytes();

                let ops = bx.debuginfo_upvar_ops_sequence(byte_offset_of_var_in_env);

                // The environment and the capture can each be indirect.
                let mut ops = if env_ref { &ops[..] } else { &ops[1..] };

                let ty = if let (true, &ty::Ref(_, ty, _)) = (by_ref, &ty.kind) {
                    ty
                } else {
                    ops = &ops[..ops.len() - 1];
                    ty
                };

                let variable_access = VariableAccess::IndirectVariable {
                    alloca: place.llval,
                    address_operations: &ops
                };
                bx.declare_local(
                    &self.debug_context,
                    name,
                    ty,
                    var_scope,
                    variable_access,
                    VariableKind::LocalVariable,
                    var_span
                );
            }
        }
    }
}
