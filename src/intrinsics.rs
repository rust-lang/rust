use crate::prelude::*;

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
        let new = $fx.bcx.ins().band(old, $src);
        $fx.bcx.ins().store(MemFlags::new(), new, $ptr, 0);
        $ret.write_cvalue($fx, CValue::ByVal(old, $fx.layout_of($T)));
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

        let ret_val = CValue::ByVal(old, $ret.layout());
        $ret.write_cvalue($fx, ret_val);
    };
}

pub fn codegen_intrinsic_call<'a, 'tcx: 'a>(
    fx: &mut FunctionCx<'a, 'tcx, impl Backend>,
    def_id: DefId,
    substs: &'tcx Substs,
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
                    trap_panic(&mut fx.bcx);
                }
                "unreachable" => {
                    trap_unreachable(&mut fx.bcx);
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
        arith_offset, (v base, v offset) {
            let res = fx.bcx.ins().iadd(base, offset);
            let res = CValue::ByVal(res, ret.layout());
            ret.write_cvalue(fx, res);
        };
        likely | unlikely, (c a) {
            ret.write_cvalue(fx, a);
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
            let discr = crate::base::trans_get_discriminant(fx, val, ret.layout());
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
                let (_ptr, info) = ptr.load_value_pair(fx);
                let (size, _align) = crate::unsize::size_and_align_of_dst(fx, layout.ty, info);
                size
            } else {
                fx
                    .bcx
                    .ins()
                    .iconst(fx.pointer_type, layout.size.bytes() as i64)
            };
            ret.write_cvalue(fx, CValue::ByVal(size, usize_layout));
        };
        min_align_of, <T> () {
            let min_align = fx.layout_of(T).align.abi.bytes();
            let min_align = CValue::const_val(fx, usize_layout.ty, min_align as i64);
            ret.write_cvalue(fx, min_align);
        };
        min_align_of_val, <T> (c ptr) {
            let layout = fx.layout_of(T);
            let align = if layout.is_unsized() {
                let (_ptr, info) = ptr.load_value_pair(fx);
                let (_size, align) = crate::unsize::size_and_align_of_dst(fx, layout.ty, info);
                align
            } else {
                fx
                    .bcx
                    .ins()
                    .iconst(fx.pointer_type, layout.align.abi.bytes() as i64)
            };
            ret.write_cvalue(fx, CValue::ByVal(align, usize_layout));
        };
        type_id, <T> () {
            let type_id = fx.tcx.type_id_hash(T);
            let type_id = CValue::const_val(fx, u64_layout.ty, type_id as i64);
            ret.write_cvalue(fx, type_id);
        };
        _ if intrinsic.starts_with("unchecked_") || intrinsic == "exact_div", (c x, c y) {
            let bin_op = match intrinsic {
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
        rotate_left, <T>(v x, v y) {
            let layout = fx.layout_of(T);
            let res = fx.bcx.ins().rotl(x, y);
            ret.write_cvalue(fx, CValue::ByVal(res, layout));
        };
        rotate_right, <T>(v x, v y) {
            let layout = fx.layout_of(T);
            let res = fx.bcx.ins().rotr(x, y);
            ret.write_cvalue(fx, CValue::ByVal(res, layout));
        };
        offset, (v base, v offset) {
            let res = fx.bcx.ins().iadd(base, offset);
            ret.write_cvalue(fx, CValue::ByVal(res, args[0].layout()));
        };
        transmute, <src_ty, dst_ty> (c from) {
            assert_eq!(from.layout().ty, src_ty);
            let addr = from.force_stack(fx);
            let dst_layout = fx.layout_of(dst_ty);
            ret.write_cvalue(fx, CValue::ByRef(addr, dst_layout))
        };
        init, <T> () {
            let layout = fx.layout_of(T);
            let inited_place = CPlace::new_stack_slot(fx, T);
            let addr = inited_place.to_addr(fx);
            let zero_val = fx.bcx.ins().iconst(types::I8, 0);
            let len_val = fx.bcx.ins().iconst(pointer_ty(fx.tcx), layout.size.bytes() as i64);
            fx.bcx.call_memset(fx.module.target_config(), addr, zero_val, len_val);

            let inited_val = inited_place.to_cvalue(fx);
            ret.write_cvalue(fx, inited_val);
        };
        write_bytes, (v dst, v val, v count) {
            fx.bcx.call_memset(fx.module.target_config(), dst, val, count);
        };
        uninit, <T> () {
            let uninit_place = CPlace::new_stack_slot(fx, T);
            let uninit_val = uninit_place.to_cvalue(fx);
            ret.write_cvalue(fx, uninit_val);
        };
        ctlz | ctlz_nonzero, <T> (v arg) {
            let res = CValue::ByVal(fx.bcx.ins().clz(arg), fx.layout_of(T));
            ret.write_cvalue(fx, res);
        };
        cttz | cttz_nonzero, <T> (v arg) {
            let res = CValue::ByVal(fx.bcx.ins().clz(arg), fx.layout_of(T));
            ret.write_cvalue(fx, res);
        };
        ctpop, <T> (v arg) {
            let res = CValue::ByVal(fx.bcx.ins().popcnt(arg), fx.layout_of(T));
            ret.write_cvalue(fx, res);
        };
        bitreverse, <T> (v arg) {
            let res = CValue::ByVal(fx.bcx.ins().bitrev(arg), fx.layout_of(T));
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
                crate::trap::trap_panic(&mut fx.bcx);
                return;
            }
        };

        _ if intrinsic.starts_with("atomic_fence"), () {};
        _ if intrinsic.starts_with("atomic_singlethreadfence"), () {};
        _ if intrinsic.starts_with("atomic_load"), (c ptr) {
            let inner_layout =
                fx.layout_of(ptr.layout().ty.builtin_deref(true).unwrap().ty);
            let val = CValue::ByRef(ptr.load_scalar(fx), inner_layout);
            ret.write_cvalue(fx, val);
        };
        _ if intrinsic.starts_with("atomic_store"), (v ptr, c val) {
            let dest = CPlace::Addr(ptr, None, val.layout());
            dest.write_cvalue(fx, val);
        };
        _ if intrinsic.starts_with("atomic_xchg"), <T> (v ptr, c src) {
            // Read old
            let clif_ty = fx.clif_type(T).unwrap();
            let old = fx.bcx.ins().load(clif_ty, MemFlags::new(), ptr, 0);
            ret.write_cvalue(fx, CValue::ByVal(old, fx.layout_of(T)));

            // Write new
            let dest = CPlace::Addr(ptr, None, src.layout());
            dest.write_cvalue(fx, src);
        };
        _ if intrinsic.starts_with("atomic_cxchg"), <T> (v ptr, v test_old, v new) { // both atomic_cxchg_* and atomic_cxchgweak_*
            // Read old
            let clif_ty = fx.clif_type(T).unwrap();
            let old = fx.bcx.ins().load(clif_ty, MemFlags::new(), ptr, 0);

            // Compare
            let is_eq = fx.bcx.ins().icmp(IntCC::Equal, old, test_old);
            let new = crate::common::codegen_select(&mut fx.bcx, is_eq, old, new); // Keep old if not equal to test_old

            // Write new
            fx.bcx.ins().store(MemFlags::new(), new, ptr, 0);

            let ret_val = CValue::ByValPair(old, fx.bcx.ins().bint(types::I8, is_eq), ret.layout());
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
            atomic_binop_return_old! (fx, bnand<T>(ptr, src) -> ret);
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
    }

    if let Some((_, dest)) = destination {
        let ret_ebb = fx.get_ebb(dest);
        fx.bcx.ins().jump(ret_ebb, &[]);
    } else {
        trap_unreachable(&mut fx.bcx);
    }
}
