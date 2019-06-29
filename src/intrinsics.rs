use crate::prelude::*;

use rustc::ty::subst::SubstsRef;

macro_rules! intrinsic_pat {
    (_) => {
        _
    };
    ($name:ident) => {
        stringify!($name)
    }
}

macro_rules! intrinsic_arg {
    (c $fx:expr, $arg:ident) => {
        $arg
    };
    (v $fx:expr, $arg:ident) => {
        $arg.load_scalar($fx)
    };
}

macro_rules! intrinsic_substs {
    ($substs:expr, $index:expr,) => {};
    ($substs:expr, $index:expr, $first:ident $(,$rest:ident)*) => {
        let $first = $substs.type_at($index);
        intrinsic_substs!($substs, $index+1, $($rest),*);
    };
}

macro_rules! intrinsic_match {
    ($fx:expr, $intrinsic:expr, $substs:expr, $args:expr, $(
        $($name:tt)|+ $(if $cond:expr)?, $(<$($subst:ident),*>)? ($($a:ident $arg:ident),*) $content:block;
    )*) => {
        match $intrinsic {
            $(
                $(intrinsic_pat!($name))|* $(if $cond)? => {
                    #[allow(unused_parens, non_snake_case)]
                    {
                        $(
                            intrinsic_substs!($substs, 0, $($subst),*);
                        )?
                        if let [$($arg),*] = *$args {
                            let ($($arg),*) = (
                                $(intrinsic_arg!($a $fx, $arg)),*
                            );
                            #[warn(unused_parens, non_snake_case)]
                            {
                                $content
                            }
                        } else {
                            bug!("wrong number of args for intrinsic {:?}", $intrinsic);
                        }
                    }
                }
            )*
            _ => unimpl!("unsupported intrinsic {}", $intrinsic),
        }
    };
}

macro_rules! atomic_binop_return_old {
    ($fx:expr, $op:ident<$T:ident>($ptr:ident, $src:ident) -> $ret:ident) => {
        let clif_ty = $fx.clif_type($T).unwrap();
        let old = $fx.bcx.ins().load(clif_ty, MemFlags::new(), $ptr, 0);
        let new = $fx.bcx.ins().$op(old, $src);
        $fx.bcx.ins().store(MemFlags::new(), new, $ptr, 0);
        $ret.write_cvalue($fx, CValue::by_val(old, $fx.layout_of($T)));
    };
}

macro_rules! atomic_minmax {
    ($fx:expr, $cc:expr, <$T:ident> ($ptr:ident, $src:ident) -> $ret:ident) => {
        // Read old
        let clif_ty = $fx.clif_type($T).unwrap();
        let old = $fx.bcx.ins().load(clif_ty, MemFlags::new(), $ptr, 0);

        // Compare
        let is_eq = $fx.bcx.ins().icmp(IntCC::SignedGreaterThan, old, $src);
        let new = crate::common::codegen_select(&mut $fx.bcx, is_eq, old, $src);

        // Write new
        $fx.bcx.ins().store(MemFlags::new(), new, $ptr, 0);

        let ret_val = CValue::by_val(old, $ret.layout());
        $ret.write_cvalue($fx, ret_val);
    };
}

pub fn codegen_intrinsic_call<'a, 'tcx: 'a>(
    fx: &mut FunctionCx<'a, 'tcx, impl Backend>,
    def_id: DefId,
    substs: SubstsRef<'tcx>,
    args: Vec<CValue<'tcx>>,
    destination: Option<(CPlace<'tcx>, BasicBlock)>,
) {
    let intrinsic = fx.tcx.item_name(def_id).as_str();
    let intrinsic = &intrinsic[..];

    let ret = match destination {
        Some((place, _)) => place,
        None => {
            // Insert non returning intrinsics here
            match intrinsic {
                "abort" => {
                    trap_panic(fx, "Called intrinsic::abort.");
                }
                "unreachable" => {
                    trap_unreachable(fx, "[corruption] Called intrinsic::unreachable.");
                }
                _ => unimplemented!("unsupported instrinsic {}", intrinsic),
            }
            return;
        }
    };

    let u64_layout = fx.layout_of(fx.tcx.types.u64);
    let usize_layout = fx.layout_of(fx.tcx.types.usize);

    intrinsic_match! {
        fx, intrinsic, substs, args,

        assume, (c _a) {};
        likely | unlikely, (c a) {
            ret.write_cvalue(fx, a);
        };
        breakpoint, () {
            fx.bcx.ins().debugtrap();
        };
        copy | copy_nonoverlapping, <elem_ty> (v src, v dst, v count) {
            let elem_size: u64 = fx.layout_of(elem_ty).size.bytes();
            let elem_size = fx
                .bcx
                .ins()
                .iconst(fx.pointer_type, elem_size as i64);
            assert_eq!(args.len(), 3);
            let byte_amount = fx.bcx.ins().imul(count, elem_size);

            if intrinsic.ends_with("_nonoverlapping") {
                fx.bcx.call_memcpy(fx.module.target_config(), dst, src, byte_amount);
            } else {
                fx.bcx.call_memmove(fx.module.target_config(), dst, src, byte_amount);
            }
        };
        discriminant_value, (c val) {
            let pointee_layout = fx.layout_of(val.layout().ty.builtin_deref(true).unwrap().ty);
            let place = CPlace::for_addr(val.load_scalar(fx), pointee_layout);
            let discr = crate::base::trans_get_discriminant(fx, place, ret.layout());
            ret.write_cvalue(fx, discr);
        };
        size_of, <T> () {
            let size_of = fx.layout_of(T).size.bytes();
            let size_of = CValue::const_val(fx, usize_layout.ty, size_of as i64);
            ret.write_cvalue(fx, size_of);
        };
        size_of_val, <T> (c ptr) {
            let layout = fx.layout_of(T);
            let size = if layout.is_unsized() {
                let (_ptr, info) = ptr.load_scalar_pair(fx);
                let (size, _align) = crate::unsize::size_and_align_of_dst(fx, layout.ty, info);
                size
            } else {
                fx
                    .bcx
                    .ins()
                    .iconst(fx.pointer_type, layout.size.bytes() as i64)
            };
            ret.write_cvalue(fx, CValue::by_val(size, usize_layout));
        };
        min_align_of, <T> () {
            let min_align = fx.layout_of(T).align.abi.bytes();
            let min_align = CValue::const_val(fx, usize_layout.ty, min_align as i64);
            ret.write_cvalue(fx, min_align);
        };
        min_align_of_val, <T> (c ptr) {
            let layout = fx.layout_of(T);
            let align = if layout.is_unsized() {
                let (_ptr, info) = ptr.load_scalar_pair(fx);
                let (_size, align) = crate::unsize::size_and_align_of_dst(fx, layout.ty, info);
                align
            } else {
                fx
                    .bcx
                    .ins()
                    .iconst(fx.pointer_type, layout.align.abi.bytes() as i64)
            };
            ret.write_cvalue(fx, CValue::by_val(align, usize_layout));
        };
        pref_align_of, <T> () {
            let pref_align = fx.layout_of(T).align.pref.bytes();
            let pref_align = CValue::const_val(fx, usize_layout.ty, pref_align as i64);
            ret.write_cvalue(fx, pref_align);
        };


        type_id, <T> () {
            let type_id = fx.tcx.type_id_hash(T);
            let type_id = CValue::const_val(fx, u64_layout.ty, type_id as i64);
            ret.write_cvalue(fx, type_id);
        };
        type_name, <T> () {
            let type_name = fx.tcx.type_name(T);
            let type_name = crate::constant::trans_const_value(fx, type_name);
            ret.write_cvalue(fx, type_name);
        };

        _ if intrinsic.starts_with("unchecked_") || intrinsic == "exact_div", (c x, c y) {
            // FIXME trap on overflow
            let bin_op = match intrinsic {
                "unchecked_sub" => BinOp::Sub,
                "unchecked_div" | "exact_div" => BinOp::Div,
                "unchecked_rem" => BinOp::Rem,
                "unchecked_shl" => BinOp::Shl,
                "unchecked_shr" => BinOp::Shr,
                _ => unimplemented!("intrinsic {}", intrinsic),
            };
            let res = match ret.layout().ty.sty {
                ty::Uint(_) => crate::base::trans_int_binop(
                    fx,
                    bin_op,
                    x,
                    y,
                    ret.layout().ty,
                    false,
                ),
                ty::Int(_) => crate::base::trans_int_binop(
                    fx,
                    bin_op,
                    x,
                    y,
                    ret.layout().ty,
                    true,
                ),
                _ => panic!(),
            };
            ret.write_cvalue(fx, res);
        };
        _ if intrinsic.ends_with("_with_overflow"), <T> (c x, c y) {
            assert_eq!(x.layout().ty, y.layout().ty);
            let bin_op = match intrinsic {
                "add_with_overflow" => BinOp::Add,
                "sub_with_overflow" => BinOp::Sub,
                "mul_with_overflow" => BinOp::Mul,
                _ => unimplemented!("intrinsic {}", intrinsic),
            };
            let res = match T.sty {
                ty::Uint(_) => crate::base::trans_checked_int_binop(
                    fx,
                    bin_op,
                    x,
                    y,
                    ret.layout().ty,
                    false,
                ),
                ty::Int(_) => crate::base::trans_checked_int_binop(
                    fx,
                    bin_op,
                    x,
                    y,
                    ret.layout().ty,
                    true,
                ),
                _ => panic!(),
            };
            ret.write_cvalue(fx, res);
        };
        _ if intrinsic.starts_with("overflowing_"), <T> (c x, c y) {
            assert_eq!(x.layout().ty, y.layout().ty);
            let bin_op = match intrinsic {
                "overflowing_add" => BinOp::Add,
                "overflowing_sub" => BinOp::Sub,
                "overflowing_mul" => BinOp::Mul,
                _ => unimplemented!("intrinsic {}", intrinsic),
            };
            let res = match T.sty {
                ty::Uint(_) => crate::base::trans_int_binop(
                    fx,
                    bin_op,
                    x,
                    y,
                    ret.layout().ty,
                    false,
                ),
                ty::Int(_) => crate::base::trans_int_binop(
                    fx,
                    bin_op,
                    x,
                    y,
                    ret.layout().ty,
                    true,
                ),
                _ => panic!(),
            };
            ret.write_cvalue(fx, res);
        };
        _ if intrinsic.starts_with("saturating_"), <T> (c x, c y) {
            // FIXME implement saturating behavior
            assert_eq!(x.layout().ty, y.layout().ty);
            let bin_op = match intrinsic {
                "saturating_add" => BinOp::Add,
                "saturating_sub" => BinOp::Sub,
                "saturating_mul" => BinOp::Mul,
                _ => unimplemented!("intrinsic {}", intrinsic),
            };
            let res = match T.sty {
                ty::Uint(_) => crate::base::trans_int_binop(
                    fx,
                    bin_op,
                    x,
                    y,
                    ret.layout().ty,
                    false,
                ),
                ty::Int(_) => crate::base::trans_int_binop(
                    fx,
                    bin_op,
                    x,
                    y,
                    ret.layout().ty,
                    true,
                ),
                _ => panic!(),
            };
            ret.write_cvalue(fx, res);
        };
        rotate_left, <T>(v x, v y) {
            let layout = fx.layout_of(T);
            let res = fx.bcx.ins().rotl(x, y);
            ret.write_cvalue(fx, CValue::by_val(res, layout));
        };
        rotate_right, <T>(v x, v y) {
            let layout = fx.layout_of(T);
            let res = fx.bcx.ins().rotr(x, y);
            ret.write_cvalue(fx, CValue::by_val(res, layout));
        };

        // The only difference between offset and arith_offset is regarding UB. Because Cranelift
        // doesn't have UB both are codegen'ed the same way
        offset | arith_offset, (c base, v offset) {
            let pointee_ty = base.layout().ty.builtin_deref(true).unwrap().ty;
            let pointee_size = fx.layout_of(pointee_ty).size.bytes();
            let ptr_diff = fx.bcx.ins().imul_imm(offset, pointee_size as i64);
            let base_val = base.load_scalar(fx);
            let res = fx.bcx.ins().iadd(base_val, ptr_diff);
            ret.write_cvalue(fx, CValue::by_val(res, args[0].layout()));
        };

        transmute, <src_ty, dst_ty> (c from) {
            assert_eq!(from.layout().ty, src_ty);
            let addr = from.force_stack(fx);
            let dst_layout = fx.layout_of(dst_ty);
            ret.write_cvalue(fx, CValue::by_ref(addr, dst_layout))
        };
        init, () {
            if ret.layout().abi == Abi::Uninhabited {
                crate::trap::trap_panic(fx, "[panic] Called intrinsic::init for uninhabited type.");
                return;
            }

            match ret {
                CPlace::NoPlace(_layout) => {}
                CPlace::Var(var, layout) => {
                    let clif_ty = fx.clif_type(layout.ty).unwrap();
                    let val = match clif_ty {
                        types::I8 | types::I16 | types::I32 | types::I64 => fx.bcx.ins().iconst(clif_ty, 0),
                        types::F32 => {
                            let zero = fx.bcx.ins().iconst(types::I32, 0);
                            fx.bcx.ins().bitcast(types::F32, zero)
                        }
                        types::F64 => {
                            let zero = fx.bcx.ins().iconst(types::I64, 0);
                            fx.bcx.ins().bitcast(types::F64, zero)
                        }
                        _ => panic!("clif_type returned {}", clif_ty),
                    };
                    fx.bcx.def_var(mir_var(var), val);
                }
                _ => {
                    let addr = ret.to_addr(fx);
                    let layout = ret.layout();
                    fx.bcx.emit_small_memset(fx.module.target_config(), addr, 0, layout.size.bytes(), 1);
                }
            }
        };
        write_bytes, (c dst, v val, v count) {
            let pointee_ty = dst.layout().ty.builtin_deref(true).unwrap().ty;
            let pointee_size = fx.layout_of(pointee_ty).size.bytes();
            let count = fx.bcx.ins().imul_imm(count, pointee_size as i64);
            let dst_ptr = dst.load_scalar(fx);
            fx.bcx.call_memset(fx.module.target_config(), dst_ptr, val, count);
        };
        ctlz | ctlz_nonzero, <T> (v arg) {
            let res = if T == fx.tcx.types.u128 || T == fx.tcx.types.i128 {
                // FIXME verify this algorithm is correct
                let (lsb, msb) = fx.bcx.ins().isplit(arg);
                let lsb_lz = fx.bcx.ins().clz(lsb);
                let msb_lz = fx.bcx.ins().clz(msb);
                let msb_lz_is_64 = fx.bcx.ins().icmp_imm(IntCC::Equal, msb_lz, 64);
                let lsb_lz_plus_64 = fx.bcx.ins().iadd_imm(lsb_lz, 64);
                fx.bcx.ins().select(msb_lz_is_64, lsb_lz_plus_64, msb_lz)
            } else {
                fx.bcx.ins().clz(arg)
            };
            let res = CValue::by_val(res, fx.layout_of(T));
            ret.write_cvalue(fx, res);
        };
        cttz | cttz_nonzero, <T> (v arg) {
            let res = if T == fx.tcx.types.u128 || T == fx.tcx.types.i128 {
                // FIXME verify this algorithm is correct
                let (lsb, msb) = fx.bcx.ins().isplit(arg);
                let lsb_tz = fx.bcx.ins().ctz(lsb);
                let msb_tz = fx.bcx.ins().ctz(msb);
                let lsb_tz_is_64 = fx.bcx.ins().icmp_imm(IntCC::Equal, lsb_tz, 64);
                let msb_lz_plus_64 = fx.bcx.ins().iadd_imm(msb_tz, 64);
                fx.bcx.ins().select(lsb_tz_is_64, msb_lz_plus_64, lsb_tz)
            } else {
                fx.bcx.ins().ctz(arg)
            };
            let res = CValue::by_val(res, fx.layout_of(T));
            ret.write_cvalue(fx, res);
        };
        ctpop, <T> (v arg) {
            let res = CValue::by_val(fx.bcx.ins().popcnt(arg), fx.layout_of(T));
            ret.write_cvalue(fx, res);
        };
        bitreverse, <T> (v arg) {
            let res = CValue::by_val(fx.bcx.ins().bitrev(arg), fx.layout_of(T));
            ret.write_cvalue(fx, res);
        };
        bswap, <T> (v arg) {
            // FIXME(CraneStation/cranelift#794) add bswap instruction to cranelift
            fn swap(bcx: &mut FunctionBuilder, v: Value) -> Value {
                match bcx.func.dfg.value_type(v) {
                    types::I8 => v,

                    // https://code.woboq.org/gcc/include/bits/byteswap.h.html
                    types::I16 => {
                        let tmp1 = bcx.ins().ishl_imm(v, 8);
                        let n1 = bcx.ins().band_imm(tmp1, 0xFF00);

                        let tmp2 = bcx.ins().ushr_imm(v, 8);
                        let n2 = bcx.ins().band_imm(tmp2, 0x00FF);

                        bcx.ins().bor(n1, n2)
                    }
                    types::I32 => {
                        let tmp1 = bcx.ins().ishl_imm(v, 24);
                        let n1 = bcx.ins().band_imm(tmp1, 0xFF00_0000);

                        let tmp2 = bcx.ins().ishl_imm(v, 8);
                        let n2 = bcx.ins().band_imm(tmp2, 0x00FF_0000);

                        let tmp3 = bcx.ins().ushr_imm(v, 8);
                        let n3 = bcx.ins().band_imm(tmp3, 0x0000_FF00);

                        let tmp4 = bcx.ins().ushr_imm(v, 24);
                        let n4 = bcx.ins().band_imm(tmp4, 0x0000_00FF);

                        let or_tmp1 = bcx.ins().bor(n1, n2);
                        let or_tmp2 = bcx.ins().bor(n3, n4);
                        bcx.ins().bor(or_tmp1, or_tmp2)
                    }
                    types::I64 => {
                        let tmp1 = bcx.ins().ishl_imm(v, 56);
                        let n1 = bcx.ins().band_imm(tmp1, 0xFF00_0000_0000_0000u64 as i64);

                        let tmp2 = bcx.ins().ishl_imm(v, 40);
                        let n2 = bcx.ins().band_imm(tmp2, 0x00FF_0000_0000_0000u64 as i64);

                        let tmp3 = bcx.ins().ishl_imm(v, 24);
                        let n3 = bcx.ins().band_imm(tmp3, 0x0000_FF00_0000_0000u64 as i64);

                        let tmp4 = bcx.ins().ishl_imm(v, 8);
                        let n4 = bcx.ins().band_imm(tmp4, 0x0000_00FF_0000_0000u64 as i64);

                        let tmp5 = bcx.ins().ushr_imm(v, 8);
                        let n5 = bcx.ins().band_imm(tmp5, 0x0000_0000_FF00_0000u64 as i64);

                        let tmp6 = bcx.ins().ushr_imm(v, 24);
                        let n6 = bcx.ins().band_imm(tmp6, 0x0000_0000_00FF_0000u64 as i64);

                        let tmp7 = bcx.ins().ushr_imm(v, 40);
                        let n7 = bcx.ins().band_imm(tmp7, 0x0000_0000_0000_FF00u64 as i64);

                        let tmp8 = bcx.ins().ushr_imm(v, 56);
                        let n8 = bcx.ins().band_imm(tmp8, 0x0000_0000_0000_00FFu64 as i64);

                        let or_tmp1 = bcx.ins().bor(n1, n2);
                        let or_tmp2 = bcx.ins().bor(n3, n4);
                        let or_tmp3 = bcx.ins().bor(n5, n6);
                        let or_tmp4 = bcx.ins().bor(n7, n8);

                        let or_tmp5 = bcx.ins().bor(or_tmp1, or_tmp2);
                        let or_tmp6 = bcx.ins().bor(or_tmp3, or_tmp4);
                        bcx.ins().bor(or_tmp5, or_tmp6)
                    }
                    types::I128 => {
                        let (lo, hi) = bcx.ins().isplit(v);
                        let lo = swap(bcx, lo);
                        let hi = swap(bcx, hi);
                        bcx.ins().iconcat(hi, lo)
                    }
                    ty => unimplemented!("bswap {}", ty),
                }
            };
            let res = CValue::by_val(swap(&mut fx.bcx, arg), fx.layout_of(T));
            ret.write_cvalue(fx, res);
        };
        needs_drop, <T> () {
            let needs_drop = if T.needs_drop(fx.tcx, ParamEnv::reveal_all()) {
                1
            } else {
                0
            };
            let needs_drop = CValue::const_val(fx, fx.tcx.types.bool, needs_drop);
            ret.write_cvalue(fx, needs_drop);
        };
        panic_if_uninhabited, <T> () {
            if fx.layout_of(T).abi.is_uninhabited() {
                crate::trap::trap_panic(fx, "[panic] Called intrinsic::panic_if_uninhabited for uninhabited type.");
                return;
            }
        };

        volatile_load, (c ptr) {
            // Cranelift treats loads as volatile by default
            let inner_layout =
                fx.layout_of(ptr.layout().ty.builtin_deref(true).unwrap().ty);
            let val = CValue::by_ref(ptr.load_scalar(fx), inner_layout);
            ret.write_cvalue(fx, val);
        };
        volatile_store, (v ptr, c val) {
            // Cranelift treats stores as volatile by default
            let dest = CPlace::for_addr(ptr, val.layout());
            dest.write_cvalue(fx, val);
        };

        _ if intrinsic.starts_with("atomic_fence"), () {};
        _ if intrinsic.starts_with("atomic_singlethreadfence"), () {};
        _ if intrinsic.starts_with("atomic_load"), (c ptr) {
            let inner_layout =
                fx.layout_of(ptr.layout().ty.builtin_deref(true).unwrap().ty);
            let val = CValue::by_ref(ptr.load_scalar(fx), inner_layout);
            ret.write_cvalue(fx, val);
        };
        _ if intrinsic.starts_with("atomic_store"), (v ptr, c val) {
            let dest = CPlace::for_addr(ptr, val.layout());
            dest.write_cvalue(fx, val);
        };
        _ if intrinsic.starts_with("atomic_xchg"), <T> (v ptr, c src) {
            // Read old
            let clif_ty = fx.clif_type(T).unwrap();
            let old = fx.bcx.ins().load(clif_ty, MemFlags::new(), ptr, 0);
            ret.write_cvalue(fx, CValue::by_val(old, fx.layout_of(T)));

            // Write new
            let dest = CPlace::for_addr(ptr, src.layout());
            dest.write_cvalue(fx, src);
        };
        _ if intrinsic.starts_with("atomic_cxchg"), <T> (v ptr, v test_old, v new) { // both atomic_cxchg_* and atomic_cxchgweak_*
            // Read old
            let clif_ty = fx.clif_type(T).unwrap();
            let old = fx.bcx.ins().load(clif_ty, MemFlags::new(), ptr, 0);

            // Compare
            let is_eq = fx.bcx.ins().icmp(IntCC::Equal, old, test_old);
            let new = crate::common::codegen_select(&mut fx.bcx, is_eq, new, old); // Keep old if not equal to test_old

            // Write new
            fx.bcx.ins().store(MemFlags::new(), new, ptr, 0);

            let ret_val = CValue::by_val_pair(old, fx.bcx.ins().bint(types::I8, is_eq), ret.layout());
            ret.write_cvalue(fx, ret_val);
        };

        _ if intrinsic.starts_with("atomic_xadd"), <T> (v ptr, v amount) {
            atomic_binop_return_old! (fx, iadd<T>(ptr, amount) -> ret);
        };
        _ if intrinsic.starts_with("atomic_xsub"), <T> (v ptr, v amount) {
            atomic_binop_return_old! (fx, isub<T>(ptr, amount) -> ret);
        };
        _ if intrinsic.starts_with("atomic_and"), <T> (v ptr, v src) {
            atomic_binop_return_old! (fx, band<T>(ptr, src) -> ret);
        };
        _ if intrinsic.starts_with("atomic_nand"), <T> (v ptr, v src) {
            atomic_binop_return_old! (fx, band_not<T>(ptr, src) -> ret);
        };
        _ if intrinsic.starts_with("atomic_or"), <T> (v ptr, v src) {
            atomic_binop_return_old! (fx, bor<T>(ptr, src) -> ret);
        };
        _ if intrinsic.starts_with("atomic_xor"), <T> (v ptr, v src) {
            atomic_binop_return_old! (fx, bxor<T>(ptr, src) -> ret);
        };

        _ if intrinsic.starts_with("atomic_max"), <T> (v ptr, v src) {
            atomic_minmax!(fx, IntCC::SignedGreaterThan, <T> (ptr, src) -> ret);
        };
        _ if intrinsic.starts_with("atomic_umax"), <T> (v ptr, v src) {
            atomic_minmax!(fx, IntCC::UnsignedGreaterThan, <T> (ptr, src) -> ret);
        };
        _ if intrinsic.starts_with("atomic_min"), <T> (v ptr, v src) {
            atomic_minmax!(fx, IntCC::SignedLessThan, <T> (ptr, src) -> ret);
        };
        _ if intrinsic.starts_with("atomic_umin"), <T> (v ptr, v src) {
            atomic_minmax!(fx, IntCC::UnsignedLessThan, <T> (ptr, src) -> ret);
        };

        expf32, (c flt) {
            let res = fx.easy_call("expf", &[flt], fx.tcx.types.f32);
            ret.write_cvalue(fx, res);
        };
        expf64, (c flt) {
            let res = fx.easy_call("exp", &[flt], fx.tcx.types.f64);
            ret.write_cvalue(fx, res);
        };
        exp2f32, (c flt) {
            let res = fx.easy_call("exp2f", &[flt], fx.tcx.types.f32);
            ret.write_cvalue(fx, res);
        };
        exp2f64, (c flt) {
            let res = fx.easy_call("exp2", &[flt], fx.tcx.types.f64);
            ret.write_cvalue(fx, res);
        };
        fabsf32, (c flt) {
            let res = fx.easy_call("fabsf", &[flt], fx.tcx.types.f32);
            ret.write_cvalue(fx, res);
        };
        fabsf64, (c flt) {
            let res = fx.easy_call("fabs", &[flt], fx.tcx.types.f64);
            ret.write_cvalue(fx, res);
        };
        sqrtf32, (c flt) {
            let res = fx.easy_call("sqrtf", &[flt], fx.tcx.types.f32);
            ret.write_cvalue(fx, res);
        };
        sqrtf64, (c flt) {
            let res = fx.easy_call("sqrt", &[flt], fx.tcx.types.f64);
            ret.write_cvalue(fx, res);
        };
        floorf32, (c flt) {
            let res = fx.easy_call("floorf", &[flt], fx.tcx.types.f32);
            ret.write_cvalue(fx, res);
        };
        floorf64, (c flt) {
            let res = fx.easy_call("floor", &[flt], fx.tcx.types.f64);
            ret.write_cvalue(fx, res);
        };
        ceilf32, (c flt) {
            let res = fx.easy_call("ceilf", &[flt], fx.tcx.types.f32);
            ret.write_cvalue(fx, res);
        };
        ceilf64, (c flt) {
            let res = fx.easy_call("ceil", &[flt], fx.tcx.types.f64);
            ret.write_cvalue(fx, res);
        };

        minnumf32, (c a, c b) {
            let res = fx.easy_call("fminf", &[a, b], fx.tcx.types.f32);
            ret.write_cvalue(fx, res);
        };
        minnumf64, (c a, c b) {
            let res = fx.easy_call("fmin", &[a, b], fx.tcx.types.f64);
            ret.write_cvalue(fx, res);
        };
        maxnumf32, (c a, c b) {
            let res = fx.easy_call("fmaxf", &[a, b], fx.tcx.types.f32);
            ret.write_cvalue(fx, res);
        };
        maxnumf64, (c a, c b) {
            let res = fx.easy_call("fmax", &[a, b], fx.tcx.types.f64);
            ret.write_cvalue(fx, res);
        };

    }

    if let Some((_, dest)) = destination {
        let ret_ebb = fx.get_ebb(dest);
        fx.bcx.ins().jump(ret_ebb, &[]);
    } else {
        trap_unreachable(fx, "[corruption] Diverging intrinsic returned.");
    }
}
