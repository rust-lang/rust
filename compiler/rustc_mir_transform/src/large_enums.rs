use rustc_abi::{HasDataLayout, Size, TagEncoding, Variants};
use rustc_data_structures::fx::FxHashMap;
use rustc_middle::mir::interpret::AllocId;
use rustc_middle::mir::*;
use rustc_middle::ty::util::IntTypeExt;
use rustc_middle::ty::{self, AdtDef, Ty, TyCtxt};
use rustc_session::Session;

use crate::patch::MirPatch;

/// A pass that seeks to optimize unnecessary moves of large enum types, if there is a large
/// enough discrepancy between them.
///
/// i.e. If there are two variants:
/// ```
/// enum Example {
///   Small,
///   Large([u32; 1024]),
/// }
/// ```
/// Instead of emitting moves of the large variant, perform a memcpy instead.
/// Based off of [this HackMD](https://hackmd.io/@ft4bxUsFT5CEUBmRKYHr7w/rJM8BBPzD).
///
/// In summary, what this does is at runtime determine which enum variant is active,
/// and instead of copying all the bytes of the largest possible variant,
/// copy only the bytes for the currently active variant.
pub(super) struct EnumSizeOpt {
    pub(crate) discrepancy: u64,
}

impl<'tcx> crate::MirPass<'tcx> for EnumSizeOpt {
    fn is_enabled(&self, sess: &Session) -> bool {
        // There are some differences in behavior on wasm and ARM that are not properly
        // understood, so we conservatively treat this optimization as unsound:
        // https://github.com/rust-lang/rust/pull/85158#issuecomment-1101836457
        sess.opts.unstable_opts.unsound_mir_opts || sess.mir_opt_level() >= 3
    }

    fn run_pass(&self, tcx: TyCtxt<'tcx>, body: &mut Body<'tcx>) {
        // NOTE: This pass may produce different MIR based on the alignment of the target
        // platform, but it will still be valid.

        let mut alloc_cache = FxHashMap::default();
        let typing_env = body.typing_env(tcx);

        let mut patch = MirPatch::new(body);

        for (block, data) in body.basic_blocks.as_mut().iter_enumerated_mut() {
            for (statement_index, st) in data.statements.iter_mut().enumerate() {
                let StatementKind::Assign(box (
                    lhs,
                    Rvalue::Use(Operand::Copy(rhs) | Operand::Move(rhs)),
                )) = &st.kind
                else {
                    continue;
                };

                let location = Location { block, statement_index };

                let ty = lhs.ty(&body.local_decls, tcx).ty;

                let Some((adt_def, num_variants, alloc_id)) =
                    self.candidate(tcx, typing_env, ty, &mut alloc_cache)
                else {
                    continue;
                };

                let span = st.source_info.span;

                let tmp_ty = Ty::new_array(tcx, tcx.types.usize, num_variants as u64);
                let size_array_local = patch.new_temp(tmp_ty, span);

                let store_live = StatementKind::StorageLive(size_array_local);

                let place = Place::from(size_array_local);
                let constant_vals = ConstOperand {
                    span,
                    user_ty: None,
                    const_: Const::Val(
                        ConstValue::Indirect { alloc_id, offset: Size::ZERO },
                        tmp_ty,
                    ),
                };
                let rval = Rvalue::Use(Operand::Constant(Box::new(constant_vals)));
                let const_assign = StatementKind::Assign(Box::new((place, rval)));

                let discr_place =
                    Place::from(patch.new_temp(adt_def.repr().discr_type().to_ty(tcx), span));
                let store_discr =
                    StatementKind::Assign(Box::new((discr_place, Rvalue::Discriminant(*rhs))));

                let discr_cast_place = Place::from(patch.new_temp(tcx.types.usize, span));
                let cast_discr = StatementKind::Assign(Box::new((
                    discr_cast_place,
                    Rvalue::Cast(CastKind::IntToInt, Operand::Copy(discr_place), tcx.types.usize),
                )));

                let size_place = Place::from(patch.new_temp(tcx.types.usize, span));
                let store_size = StatementKind::Assign(Box::new((
                    size_place,
                    Rvalue::Use(Operand::Copy(Place {
                        local: size_array_local,
                        projection: tcx.mk_place_elems(&[PlaceElem::Index(discr_cast_place.local)]),
                    })),
                )));

                let dst = Place::from(patch.new_temp(Ty::new_mut_ptr(tcx, ty), span));
                let dst_ptr =
                    StatementKind::Assign(Box::new((dst, Rvalue::RawPtr(RawPtrKind::Mut, *lhs))));

                let dst_cast_ty = Ty::new_mut_ptr(tcx, tcx.types.u8);
                let dst_cast_place = Place::from(patch.new_temp(dst_cast_ty, span));
                let dst_cast = StatementKind::Assign(Box::new((
                    dst_cast_place,
                    Rvalue::Cast(CastKind::PtrToPtr, Operand::Copy(dst), dst_cast_ty),
                )));

                let src = Place::from(patch.new_temp(Ty::new_imm_ptr(tcx, ty), span));
                let src_ptr =
                    StatementKind::Assign(Box::new((src, Rvalue::RawPtr(RawPtrKind::Const, *rhs))));

                let src_cast_ty = Ty::new_imm_ptr(tcx, tcx.types.u8);
                let src_cast_place = Place::from(patch.new_temp(src_cast_ty, span));
                let src_cast = StatementKind::Assign(Box::new((
                    src_cast_place,
                    Rvalue::Cast(CastKind::PtrToPtr, Operand::Copy(src), src_cast_ty),
                )));

                let deinit_old = StatementKind::Deinit(Box::new(dst));

                let copy_bytes = StatementKind::Intrinsic(Box::new(
                    NonDivergingIntrinsic::CopyNonOverlapping(CopyNonOverlapping {
                        src: Operand::Copy(src_cast_place),
                        dst: Operand::Copy(dst_cast_place),
                        count: Operand::Copy(size_place),
                    }),
                ));

                let store_dead = StatementKind::StorageDead(size_array_local);

                let stmts = [
                    store_live,
                    const_assign,
                    store_discr,
                    cast_discr,
                    store_size,
                    dst_ptr,
                    dst_cast,
                    src_ptr,
                    src_cast,
                    deinit_old,
                    copy_bytes,
                    store_dead,
                ];
                for stmt in stmts {
                    patch.add_statement(location, stmt);
                }

                st.make_nop(true);
            }
        }

        patch.apply(body);
    }

    fn is_required(&self) -> bool {
        false
    }
}

impl EnumSizeOpt {
    fn candidate<'tcx>(
        &self,
        tcx: TyCtxt<'tcx>,
        typing_env: ty::TypingEnv<'tcx>,
        ty: Ty<'tcx>,
        alloc_cache: &mut FxHashMap<Ty<'tcx>, AllocId>,
    ) -> Option<(AdtDef<'tcx>, usize, AllocId)> {
        let adt_def = match ty.kind() {
            ty::Adt(adt_def, _args) if adt_def.is_enum() => adt_def,
            _ => return None,
        };
        let layout = tcx.layout_of(typing_env.as_query_input(ty)).ok()?;
        let variants = match &layout.variants {
            Variants::Single { .. } | Variants::Empty => return None,
            Variants::Multiple { tag_encoding: TagEncoding::Niche { .. }, .. } => return None,

            Variants::Multiple { variants, .. } if variants.len() <= 1 => return None,
            Variants::Multiple { variants, .. } => variants,
        };
        let min = variants.iter().map(|v| v.size).min().unwrap();
        let max = variants.iter().map(|v| v.size).max().unwrap();
        if max.bytes() - min.bytes() < self.discrepancy {
            return None;
        }

        let num_discrs = adt_def.discriminants(tcx).count();
        if variants.iter_enumerated().any(|(var_idx, _)| {
            let discr_for_var = adt_def.discriminant_for_variant(tcx, var_idx).val;
            (discr_for_var > usize::MAX as u128) || (discr_for_var as usize >= num_discrs)
        }) {
            return None;
        }
        if let Some(alloc_id) = alloc_cache.get(&ty) {
            return Some((*adt_def, num_discrs, *alloc_id));
        }

        let data_layout = tcx.data_layout();
        let ptr_sized_int = data_layout.ptr_sized_integer();
        let target_bytes = ptr_sized_int.size().bytes() as usize;
        let mut data = vec![0; target_bytes * num_discrs];

        // We use a macro because `$bytes` can be u32 or u64.
        macro_rules! encode_store {
            ($curr_idx: expr, $endian: expr, $bytes: expr) => {
                let bytes = match $endian {
                    rustc_abi::Endian::Little => $bytes.to_le_bytes(),
                    rustc_abi::Endian::Big => $bytes.to_be_bytes(),
                };
                for (i, b) in bytes.into_iter().enumerate() {
                    data[$curr_idx + i] = b;
                }
            };
        }

        for (var_idx, layout) in variants.iter_enumerated() {
            let curr_idx =
                target_bytes * adt_def.discriminant_for_variant(tcx, var_idx).val as usize;
            let sz = layout.size;
            match ptr_sized_int {
                rustc_abi::Integer::I32 => {
                    encode_store!(curr_idx, data_layout.endian, sz.bytes() as u32);
                }
                rustc_abi::Integer::I64 => {
                    encode_store!(curr_idx, data_layout.endian, sz.bytes());
                }
                _ => unreachable!(),
            };
        }
        let alloc = interpret::Allocation::from_bytes(
            data,
            tcx.data_layout.ptr_sized_integer().align(&tcx.data_layout).abi,
            Mutability::Not,
            (),
        );
        let alloc = tcx.reserve_and_set_memory_alloc(tcx.mk_const_alloc(alloc));
        Some((*adt_def, num_discrs, *alloc_cache.entry(ty).or_insert(alloc)))
    }
}
