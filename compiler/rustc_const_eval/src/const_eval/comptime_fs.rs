use std::io::ErrorKind;

use rustc_abi::FieldIdx;
use rustc_errors::Diag;
use rustc_hir::def_id::LOCAL_CRATE;
use rustc_middle::comptime::ReadFileError;
use rustc_middle::mir::interpret::{
    CtfeProvenance, InterpResult, ReportedErrorInfo, Scalar, interp_ok,
};
use rustc_middle::query::LocalCrate;
use rustc_middle::ty::layout::HasTyCtxt;
use rustc_middle::ty::{Ty, TyCtxt};
use rustc_middle::{mir, throw_inval, ty};
use rustc_span::{Symbol, sym};
use tracing::instrument;

use crate::const_eval::CompileTimeMachine;
use crate::interpret::{self, Immediate, InterpCx, OpTy, PlaceTy, Projectable};

#[instrument(skip(tcx), level = "debug")]
pub fn experimental_comptime_list_files_provider(tcx: TyCtxt<'_>, _: LocalCrate) -> &[Symbol] {
    let filelist = tcx
        .sess
        .opts
        .unstable_opts
        .experimental_comptime_files
        .iter()
        .map(|p| Symbol::intern(p))
        .collect::<Vec<_>>();

    tcx.arena.alloc_slice(&filelist)
}

#[instrument(skip(tcx), level = "debug")]
pub fn experimental_comptime_fetch_file_provider<'tcx>(
    tcx: TyCtxt<'tcx>,
    key: Symbol,
) -> Result<Option<&'tcx [u8]>, ReadFileError<'tcx>> {
    // In case we are in a context without a crate root, just return that
    // every file is whitelisted but does not exist. Is this code path
    // taken by rustdoc when checking a comment code block?
    let Some(crate_root_path) = tcx.sess.io.input.opt_path() else {
        return Ok(None);
    };

    let requested_path = key.as_str();

    // Always resolve path from the crate root to provide consistent behavior.
    let resolved_path = crate_root_path.with_file_name(requested_path);

    // Very simple implementation where we whitelist paths directly. We don't even attempt
    // to canonicalize for now.
    if tcx
        .experimental_comptime_list_files(LOCAL_CRATE)
        .iter()
        .find(|s| requested_path == s.as_str())
        .is_some()
    {
        match std::fs::read(resolved_path) {
            Ok(content) => Ok(Some(tcx.arena.alloc_slice(&content))),
            Err(e) if e.kind() == ErrorKind::NotFound => Ok(None),
            Err(e) => Err(ReadFileError::IO(tcx.arena.alloc_str(&e.to_string()))),
        }
    } else {
        Err(ReadFileError::NotWhitelisted)
    }
}

impl<'tcx> InterpCx<'tcx, CompileTimeMachine<'tcx>> {
    fn build_fs_diag(&self, msg: String) -> Diag<'_> {
        let parent_span =
            self.stack().iter().rev().nth(1).map(|parent_frame| parent_frame.current_span());

        let item_span = self.tcx.span;

        self.tcx.dcx().struct_err(msg).with_span(parent_span).with_span_context(item_span)
    }

    /// intrinsics related to `comptime_ops_*`
    pub(crate) fn call_comptime_ops_intrinsic(
        &mut self,
        intrinsic_name: Symbol,
        _instance: ty::Instance<'tcx>,
        args: &[OpTy<'tcx>],
        dest: &PlaceTy<'tcx, CtfeProvenance>,
        target: Option<mir::BasicBlock>,
    ) -> InterpResult<'tcx, bool> {
        match intrinsic_name {
            sym::comptime_fetch_file => {
                let arg_place = self.deref_pointer(&args[0])?;
                let path = self.read_str(&arg_place)?;
                let path_sym = Symbol::intern(path);

                let krate = self
                    .stack()
                    .first()
                    .expect("should have a call frame if we are here")
                    .instance()
                    .def_id()
                    .krate;

                let content_maybe = match self
                    .tcx
                    .experimental_comptime_fetch_file((krate, path_sym))
                {
                    Ok(content) => content,
                    Err(ReadFileError::IO(err)) => {
                        let guard = self
                            .build_fs_diag(format!("couldn't read `{path_sym}`: {err}"))
                            .emit_err();
                        throw_inval!(AlreadyReported(ReportedErrorInfo::allowed_in_infallible(
                            guard
                        )));
                    }

                    Err(ReadFileError::NotWhitelisted) => {
                        let guard = self
                            .build_fs_diag(format!("not allowed to read `{path_sym}`: not whitelisted"))
                            .with_help("pass -Zexperimental-comptime-files=<path> to the compiler to allow reading this file")
                            .emit_err();
                        throw_inval!(AlreadyReported(ReportedErrorInfo::allowed_in_infallible(
                            guard
                        )));
                    }
                };

                let variant_index = if let Some(value) = content_maybe {
                    let ptr = self.allocate_bytes_dedup(value)?;

                    let (variant_idx, variant_place) =
                        self.project_downcast_named(dest, sym::Some)?;
                    let field_place = self.project_field(&variant_place, FieldIdx::ZERO)?;
                    self.write_immediate(
                        Immediate::ScalarPair(
                            Scalar::from_pointer(ptr, self),
                            Scalar::from_target_usize(value.len() as u64, self),
                        ),
                        &field_place,
                    )?;

                    variant_idx
                } else {
                    self.project_downcast_named(dest, sym::None)?.0
                };

                self.write_discriminant(variant_index, dest)?;
            }

            sym::comptime_list_files => {
                let krate = self
                    .stack()
                    .first()
                    .expect("should have a call frame if we are here")
                    .instance()
                    .def_id()
                    .krate;

                let filelist = self.tcx().experimental_comptime_list_files(krate);

                let element_type = dest
                    .layout()
                    .ty
                    .builtin_deref(false)
                    .unwrap()
                    .sequence_element_type(self.tcx());

                // type and layout of [&str, filelist.len()];
                let array_ty =
                    self.layout_of(Ty::new_array(self.tcx(), element_type, filelist.len() as _))?;

                // actual allocated array
                let array_place = self.allocate(array_ty, interpret::MemoryKind::Stack)?;

                let mut array_iter = self.project_array_fields(&array_place)?;
                while let Some((idx, entry_place)) = array_iter.next(self)? {
                    let file = filelist[idx as usize];
                    let file_place = self.allocate_str_dedup(file.as_str())?;
                    let imm = file_place.to_ref(self);
                    self.write_immediate(imm, &entry_place)?;
                }

                let array_place = array_place.map_provenance(CtfeProvenance::as_immutable);
                self.write_immediate(
                    Immediate::new_slice(array_place.ptr(), filelist.len() as _, self),
                    dest,
                )?;
            }

            _ => return interp_ok(false),
        }

        self.return_to_block(target)?;
        interp_ok(true)
    }
}
