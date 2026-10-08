//! Lock-free atomics for the sizes GCC lowers to libatomic calls even where LLVM inlines them,
//! through the `__sync_*` builtins: GCC still inlines those, but they are always SeqCst.

use gccjit::{BinaryOp, ComparisonOp, RValue, ToRValue, Type, UnaryOp};
use rustc_abi::Size;
use rustc_codegen_ssa::common::AtomicRmwBinOp;
use rustc_codegen_ssa::traits::BuilderMethods;
use rustc_middle::ty::layout::HasTyCtxt;
use rustc_span::Symbol;
use rustc_target::spec::Arch;

use crate::builder::Builder;
use crate::common::SignType;

impl<'a, 'gcc, 'tcx> Builder<'a, 'gcc, 'tcx> {
    /// GCC turns every 16-byte `__atomic_*` builtin into a libatomic call, even where LLVM inlines a
    /// lock-free sequence. The always-SeqCst `__sync_*` builtins are still inlined there.
    pub fn use_sync_atomics(&self, size: u64) -> bool {
        let sess = self.tcx().sess;
        size == 16
            && self.supports_128bit_integers
            && match sess.target.arch {
                Arch::AArch64 => true,
                Arch::X86_64 => {
                    sess.internal_target_features.contains(&Symbol::intern("cmpxchg16b"))
                }
                _ => false,
            }
    }

    pub fn sync_atomic_load(
        &mut self,
        typ: Type<'gcc>,
        ptr: RValue<'gcc>,
        size: Size,
    ) -> RValue<'gcc> {
        let zero = self.context.new_rvalue_zero(self.type_ix(size.bits()));
        let value = self.sync_compare_and_swap(ptr, zero, zero, size.bytes());
        self.context.new_cast(self.location, value, typ)
    }

    pub fn sync_atomic_store(&mut self, value: RValue<'gcc>, ptr: RValue<'gcc>, size: Size) {
        self.sync_compare_and_swap_loop(ptr, value.get_type(), size, |builder, current| {
            builder.context.new_cast(builder.location, value, current.get_type())
        });
    }

    /// Returns the previous value and whether the swap happened.
    pub fn sync_atomic_cmpxchg(
        &mut self,
        dst: RValue<'gcc>,
        cmp: RValue<'gcc>,
        src: RValue<'gcc>,
        size: u64,
    ) -> (RValue<'gcc>, RValue<'gcc>) {
        let func = self.current_func();
        let expected = self.new_temp(func, self.location, cmp.get_type());
        self.llbb().add_assignment(self.location, expected, cmp);
        let previous = self.sync_compare_and_swap(dst, expected.to_rvalue(), src, size);
        let previous = self.context.new_cast(self.location, previous, cmp.get_type());
        let success = self.new_temp(func, self.location, self.bool_type);
        let comparison = self.context.new_comparison(
            self.location,
            ComparisonOp::Equals,
            previous,
            expected.to_rvalue(),
        );
        self.llbb().add_assignment(self.location, success, comparison);
        (previous, success.to_rvalue())
    }

    pub fn sync_atomic_rmw(
        &mut self,
        op: AtomicRmwBinOp,
        dst: RValue<'gcc>,
        src: RValue<'gcc>,
        size: Size,
    ) -> RValue<'gcc> {
        let src_type = src.get_type();
        let name = match op {
            AtomicRmwBinOp::AtomicAdd => "add",
            AtomicRmwBinOp::AtomicSub => "sub",
            AtomicRmwBinOp::AtomicAnd => "and",
            AtomicRmwBinOp::AtomicOr => "or",
            AtomicRmwBinOp::AtomicXor => "xor",
            AtomicRmwBinOp::AtomicXchg => {
                return self.sync_compare_and_swap_loop(dst, src_type, size, |builder, current| {
                    builder.context.new_cast(builder.location, src, current.get_type())
                });
            }
            // `__sync_fetch_and_nand` emits a note that crashes libgccjit's diagnostic printer.
            AtomicRmwBinOp::AtomicNand => {
                return self.sync_compare_and_swap_loop(dst, src_type, size, |builder, current| {
                    let int_type = current.get_type();
                    let src = builder.context.new_cast(builder.location, src, int_type);
                    let and = builder.context.new_binary_op(
                        builder.location,
                        BinaryOp::BitwiseAnd,
                        int_type,
                        current,
                        src,
                    );
                    builder.context.new_unary_op(
                        builder.location,
                        UnaryOp::BitwiseNegate,
                        int_type,
                        and,
                    )
                });
            }
            AtomicRmwBinOp::AtomicMax => {
                return self.sync_atomic_extremum(
                    dst,
                    src,
                    size,
                    src_type.to_signed(self),
                    ComparisonOp::GreaterThanEquals,
                );
            }
            AtomicRmwBinOp::AtomicMin => {
                return self.sync_atomic_extremum(
                    dst,
                    src,
                    size,
                    src_type.to_signed(self),
                    ComparisonOp::LessThanEquals,
                );
            }
            AtomicRmwBinOp::AtomicUMax => {
                return self.sync_atomic_extremum(
                    dst,
                    src,
                    size,
                    src_type.to_unsigned(self),
                    ComparisonOp::GreaterThanEquals,
                );
            }
            AtomicRmwBinOp::AtomicUMin => {
                return self.sync_atomic_extremum(
                    dst,
                    src,
                    size,
                    src_type.to_unsigned(self),
                    ComparisonOp::LessThanEquals,
                );
            }
        };

        let fetch_and_op = self.context.get_builtin_function(format!(
            "__sync_fetch_and_{}_{}",
            name,
            size.bytes()
        ));
        let pointer_type = fetch_and_op.get_param(0).to_rvalue().get_type();
        let dst = self.context.new_cast(self.location, dst, pointer_type);
        let int_type = fetch_and_op.get_param(1).to_rvalue().get_type();
        let src = self.context.new_cast(self.location, src, int_type);
        let result = self.context.new_call(self.location, fetch_and_op, &[dst, src]);
        self.context.new_cast(self.location, result, src_type)
    }

    /// Min/max: keeps the current value if `current keep_current_operator src` holds, with both
    /// compared as `comparison_type`; stores `src` otherwise.
    fn sync_atomic_extremum(
        &mut self,
        dst: RValue<'gcc>,
        src: RValue<'gcc>,
        size: Size,
        comparison_type: Type<'gcc>,
        keep_current_operator: ComparisonOp,
    ) -> RValue<'gcc> {
        self.sync_compare_and_swap_loop(dst, src.get_type(), size, |builder, current| {
            let keep_current = builder.context.new_comparison(
                builder.location,
                keep_current_operator,
                builder.context.new_cast(builder.location, current, comparison_type),
                builder.context.new_cast(builder.location, src, comparison_type),
            );
            let src = builder.context.new_cast(builder.location, src, current.get_type());
            builder.select(keep_current, current, src)
        })
    }

    /// Returns the value `dst` held before the operation.
    fn sync_compare_and_swap(
        &mut self,
        dst: RValue<'gcc>,
        expected: RValue<'gcc>,
        desired: RValue<'gcc>,
        size: u64,
    ) -> RValue<'gcc> {
        let compare_and_swap =
            self.context.get_builtin_function(format!("__sync_val_compare_and_swap_{}", size));
        let pointer_type = compare_and_swap.get_param(0).to_rvalue().get_type();
        let dst = self.context.new_cast(self.location, dst, pointer_type);
        let int_type = compare_and_swap.get_param(1).to_rvalue().get_type();
        let expected = self.context.new_cast(self.location, expected, int_type);
        let desired = self.context.new_cast(self.location, desired, int_type);
        // A temporary keeps the call from being evaluated again wherever the result is used.
        let previous = self.new_temp(self.current_func(), self.location, int_type);
        let call =
            self.context.new_call(self.location, compare_and_swap, &[dst, expected, desired]);
        self.llbb().add_assignment(self.location, previous, call);
        previous.to_rvalue()
    }

    /// Stores `new_value(current)` with a compare-and-swap loop and returns the replaced value.
    fn sync_compare_and_swap_loop(
        &mut self,
        dst: RValue<'gcc>,
        typ: Type<'gcc>,
        size: Size,
        new_value: impl FnOnce(&mut Self, RValue<'gcc>) -> RValue<'gcc>,
    ) -> RValue<'gcc> {
        let func = self.current_func();
        let int_type = self.type_ix(size.bits());
        let current = self.new_temp(func, self.location, int_type);
        // Start from 0 instead of loading first: if `dst` holds something else, the first swap
        // fails and returns the actual value, which the next iteration uses.
        self.llbb().add_assignment(self.location, current, self.context.new_rvalue_zero(int_type));

        let loop_block = func.new_block("sync_compare_and_swap_loop");
        let after_block = func.new_block("after_sync_compare_and_swap_loop");
        self.llbb().end_with_jump(self.location, loop_block);
        self.switch_to_block(loop_block);

        let desired = new_value(self, current.to_rvalue());
        let previous = self.sync_compare_and_swap(dst, current.to_rvalue(), desired, size.bytes());
        let previous = self.context.new_cast(self.location, previous, int_type);
        let swapped = self.new_temp(func, self.location, self.bool_type);
        let comparison = self.context.new_comparison(
            self.location,
            ComparisonOp::Equals,
            previous,
            current.to_rvalue(),
        );
        self.llbb().add_assignment(self.location, swapped, comparison);
        self.llbb().add_assignment(self.location, current, previous);
        self.llbb().end_with_conditional(
            self.location,
            swapped.to_rvalue(),
            after_block,
            loop_block,
        );
        self.switch_to_block(after_block);

        self.context.new_cast(self.location, current.to_rvalue(), typ)
    }
}
