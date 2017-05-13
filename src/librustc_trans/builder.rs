// Copyright 2013 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#![allow(dead_code)] // FFI wrappers

use llvm;
use llvm::{AtomicRmwBinOp, AtomicOrdering, SynchronizationScope, AsmDialect};
use llvm::{Opcode, IntPredicate, RealPredicate, False, OperandBundleDef};
use llvm::{ValueRef, BasicBlockRef, BuilderRef, ModuleRef};
use common::*;
use machine::llalign_of_pref;
use type_::Type;
use value::Value;
use libc::{c_uint, c_char};
use rustc::ty::{Ty, TyCtxt, TypeFoldable};
use rustc::session::Session;
use type_of;

use std::borrow::Cow;
use std::ffi::CString;
use std::ptr;
use syntax_pos::Span;

// All Builders must have an llfn associated with them
#[must_use]
pub struct Builder<'a, 'tcx: 'a> {
    pub llbuilder: BuilderRef,
    pub ccx: &'a CrateContext<'a, 'tcx>,
}

impl<'a, 'tcx> Drop for Builder<'a, 'tcx> {
    fn drop(&mut self) {
        unsafe {
            llvm::LLVMDisposeBuilder(self.llbuilder);
        }
    }
}

// This is a really awful way to get a zero-length c-string, but better (and a
// lot more efficient) than doing str::as_c_str("", ...) every time.
fn noname() -> *const c_char {
    static CNULL: c_char = 0;
    &CNULL
}

impl<'a, 'tcx> Builder<'a, 'tcx> {
    pub fn new_block<'b>(ccx: &'a CrateContext<'a, 'tcx>, llfn: ValueRef, name: &'b str) -> Self {
        let builder = Builder::with_ccx(ccx);
        let llbb = unsafe {
            let name = CString::new(name).unwrap();
            llvm::LLVMAppendBasicBlockInContext(
                ccx.llcx(),
                llfn,
                name.as_ptr()
            )
        };
        builder.position_at_end(llbb);
        builder
    }

    pub fn with_ccx(ccx: &'a CrateContext<'a, 'tcx>) -> Self {
        // Create a fresh builder from the crate context.
        let llbuilder = unsafe {
            llvm::LLVMCreateBuilderInContext(ccx.llcx())
        };
        Builder {
            llbuilder: llbuilder,
            ccx: ccx,
        }
    }

    pub fn build_sibling_block<'b>(&self, name: &'b str) -> Builder<'a, 'tcx> {
        Builder::new_block(self.ccx, self.llfn(), name)
    }

    pub fn sess(&self) -> &Session {
        self.ccx.sess()
    }

    pub fn tcx(&self) -> TyCtxt<'a, 'tcx, 'tcx> {
        self.ccx.tcx()
    }

    pub fn llfn(&self) -> ValueRef {
        unsafe {
            llvm::LLVMGetBasicBlockParent(self.llbb())
        }
    }

    pub fn llbb(&self) -> BasicBlockRef {
        unsafe {
            llvm::LLVMGetInsertBlock(self.llbuilder)
        }
    }

    fn count_insn(&self, category: &str) {
        if self.ccx.sess().trans_stats() {
            self.ccx.stats().n_llvm_insns.set(self.ccx.stats().n_llvm_insns.get() + 1);
        }
        if self.ccx.sess().count_llvm_insns() {
            let mut h = self.ccx.stats().llvm_insns.borrow_mut();
            *h.entry(category.to_string()).or_insert(0) += 1;
        }
    }

    pub fn position_before(&self, insn: ValueRef) {
        unsafe {
            llvm::LLVMPositionBuilderBefore(self.llbuilder, insn);
        }
    }

    pub fn position_at_end(&self, llbb: BasicBlockRef) {
        unsafe {
            llvm::LLVMPositionBuilderAtEnd(self.llbuilder, llbb);
        }
    }

    pub fn position_at_start(&self, llbb: BasicBlockRef) {
        unsafe {
            llvm::LLVMRustPositionBuilderAtStart(self.llbuilder, llbb);
        }
    }

    pub fn ret_void(&self) {
        self.count_insn("retvoid");
        unsafe {
            llvm::LLVMBuildRetVoid(self.llbuilder);
        }
    }

    pub fn ret(&self, v: ValueRef) {
        self.count_insn("ret");
        unsafe {
            llvm::LLVMBuildRet(self.llbuilder, v);
        }
    }

    pub fn aggregate_ret(&self, ret_vals: &[ValueRef]) {
        unsafe {
            llvm::LLVMBuildAggregateRet(self.llbuilder,
                                        ret_vals.as_ptr(),
                                        ret_vals.len() as c_uint);
        }
    }

    pub fn br(&self, dest: BasicBlockRef) {
        self.count_insn("br");
        unsafe {
            llvm::LLVMBuildBr(self.llbuilder, dest);
        }
    }

    pub fn cond_br(&self, cond: ValueRef, then_llbb: BasicBlockRef, else_llbb: BasicBlockRef) {
        self.count_insn("condbr");
        unsafe {
            llvm::LLVMBuildCondBr(self.llbuilder, cond, then_llbb, else_llbb);
        }
    }

    pub fn switch(&self, v: ValueRef, else_llbb: BasicBlockRef, num_cases: usize) -> ValueRef {
        unsafe {
            llvm::LLVMBuildSwitch(self.llbuilder, v, else_llbb, num_cases as c_uint)
        }
    }

    pub fn indirect_br(&self, addr: ValueRef, num_dests: usize) {
        self.count_insn("indirectbr");
        unsafe {
            llvm::LLVMBuildIndirectBr(self.llbuilder, addr, num_dests as c_uint);
        }
    }

    pub fn invoke(&self,
                  llfn: ValueRef,
                  args: &[ValueRef],
                  then: BasicBlockRef,
                  catch: BasicBlockRef,
                  bundle: Option<&OperandBundleDef>) -> ValueRef {
        self.count_insn("invoke");

        debug!("Invoke {:?} with args ({})",
               Value(llfn),
               args.iter()
                   .map(|&v| format!("{:?}", Value(v)))
                   .collect::<Vec<String>>()
                   .join(", "));

        let args = self.check_call("invoke", llfn, args);
        let bundle = bundle.as_ref().map(|b| b.raw()).unwrap_or(ptr::null_mut());

        unsafe {
            llvm::LLVMRustBuildInvoke(self.llbuilder,
                                      llfn,
                                      args.as_ptr(),
                                      args.len() as c_uint,
                                      then,
                                      catch,
                                      bundle,
                                      noname())
        }
    }

    pub fn unreachable(&self) {
        self.count_insn("unreachable");
        unsafe {
            llvm::LLVMBuildUnreachable(self.llbuilder);
        }
    }

    /* Arithmetic */
    pub fn add(&self, lhs: ValueRef, rhs: ValueRef) -> ValueRef {
        self.count_insn("add");
        unsafe {
            llvm::LLVMBuildAdd(self.llbuilder, lhs, rhs, noname())
        }
    }

    pub fn nswadd(&self, lhs: ValueRef, rhs: ValueRef) -> ValueRef {
        self.count_insn("nswadd");
        unsafe {
            llvm::LLVMBuildNSWAdd(self.llbuilder, lhs, rhs, noname())
        }
    }

    pub fn nuwadd(&self, lhs: ValueRef, rhs: ValueRef) -> ValueRef {
        self.count_insn("nuwadd");
        unsafe {
            llvm::LLVMBuildNUWAdd(self.llbuilder, lhs, rhs, noname())
        }
    }

    pub fn fadd(&self, lhs: ValueRef, rhs: ValueRef) -> ValueRef {
        self.count_insn("fadd");
        unsafe {
            llvm::LLVMBuildFAdd(self.llbuilder, lhs, rhs, noname())
        }
    }

    pub fn fadd_fast(&self, lhs: ValueRef, rhs: ValueRef) -> ValueRef {
        self.count_insn("fadd");
        unsafe {
            let instr = llvm::LLVMBuildFAdd(self.llbuilder, lhs, rhs, noname());
            llvm::LLVMRustSetHasUnsafeAlgebra(instr);
            instr
        }
    }

    pub fn sub(&self, lhs: ValueRef, rhs: ValueRef) -> ValueRef {
        self.count_insn("sub");
        unsafe {
            llvm::LLVMBuildSub(self.llbuilder, lhs, rhs, noname())
        }
    }

    pub fn nswsub(&self, lhs: ValueRef, rhs: ValueRef) -> ValueRef {
        self.count_insn("nwsub");
        unsafe {
            llvm::LLVMBuildNSWSub(self.llbuilder, lhs, rhs, noname())
        }
    }

    pub fn nuwsub(&self, lhs: ValueRef, rhs: ValueRef) -> ValueRef {
        self.count_insn("nuwsub");
        unsafe {
            llvm::LLVMBuildNUWSub(self.llbuilder, lhs, rhs, noname())
        }
    }

    pub fn fsub(&self, lhs: ValueRef, rhs: ValueRef) -> ValueRef {
        self.count_insn("sub");
        unsafe {
            llvm::LLVMBuildFSub(self.llbuilder, lhs, rhs, noname())
        }
    }

    pub fn fsub_fast(&self, lhs: ValueRef, rhs: ValueRef) -> ValueRef {
        self.count_insn("sub");
        unsafe {
            let instr = llvm::LLVMBuildFSub(self.llbuilder, lhs, rhs, noname());
            llvm::LLVMRustSetHasUnsafeAlgebra(instr);
            instr
        }
    }

    pub fn mul(&self, lhs: ValueRef, rhs: ValueRef) -> ValueRef {
        self.count_insn("mul");
        unsafe {
            llvm::LLVMBuildMul(self.llbuilder, lhs, rhs, noname())
        }
    }

    pub fn nswmul(&self, lhs: ValueRef, rhs: ValueRef) -> ValueRef {
        self.count_insn("nswmul");
        unsafe {
            llvm::LLVMBuildNSWMul(self.llbuilder, lhs, rhs, noname())
        }
    }

    pub fn nuwmul(&self, lhs: ValueRef, rhs: ValueRef) -> ValueRef {
        self.count_insn("nuwmul");
        unsafe {
            llvm::LLVMBuildNUWMul(self.llbuilder, lhs, rhs, noname())
        }
    }

    pub fn fmul(&self, lhs: ValueRef, rhs: ValueRef) -> ValueRef {
        self.count_insn("fmul");
        unsafe {
            llvm::LLVMBuildFMul(self.llbuilder, lhs, rhs, noname())
        }
    }

    pub fn fmul_fast(&self, lhs: ValueRef, rhs: ValueRef) -> ValueRef {
        self.count_insn("fmul");
        unsafe {
            let instr = llvm::LLVMBuildFMul(self.llbuilder, lhs, rhs, noname());
            llvm::LLVMRustSetHasUnsafeAlgebra(instr);
            instr
        }
    }


    pub fn udiv(&self, lhs: ValueRef, rhs: ValueRef) -> ValueRef {
        self.count_insn("udiv");
        unsafe {
            llvm::LLVMBuildUDiv(self.llbuilder, lhs, rhs, noname())
        }
    }

    pub fn sdiv(&self, lhs: ValueRef, rhs: ValueRef) -> ValueRef {
        self.count_insn("sdiv");
        unsafe {
            llvm::LLVMBuildSDiv(self.llbuilder, lhs, rhs, noname())
        }
    }

    pub fn exactsdiv(&self, lhs: ValueRef, rhs: ValueRef) -> ValueRef {
        self.count_insn("exactsdiv");
        unsafe {
            llvm::LLVMBuildExactSDiv(self.llbuilder, lhs, rhs, noname())
        }
    }

    pub fn fdiv(&self, lhs: ValueRef, rhs: ValueRef) -> ValueRef {
        self.count_insn("fdiv");
        unsafe {
            llvm::LLVMBuildFDiv(self.llbuilder, lhs, rhs, noname())
        }
    }

    pub fn fdiv_fast(&self, lhs: ValueRef, rhs: ValueRef) -> ValueRef {
        self.count_insn("fdiv");
        unsafe {
            let instr = llvm::LLVMBuildFDiv(self.llbuilder, lhs, rhs, noname());
            llvm::LLVMRustSetHasUnsafeAlgebra(instr);
            instr
        }
    }

    pub fn urem(&self, lhs: ValueRef, rhs: ValueRef) -> ValueRef {
        self.count_insn("urem");
        unsafe {
            llvm::LLVMBuildURem(self.llbuilder, lhs, rhs, noname())
        }
    }

    pub fn srem(&self, lhs: ValueRef, rhs: ValueRef) -> ValueRef {
        self.count_insn("srem");
        unsafe {
            llvm::LLVMBuildSRem(self.llbuilder, lhs, rhs, noname())
        }
    }

    pub fn frem(&self, lhs: ValueRef, rhs: ValueRef) -> ValueRef {
        self.count_insn("frem");
        unsafe {
            llvm::LLVMBuildFRem(self.llbuilder, lhs, rhs, noname())
        }
    }

    pub fn frem_fast(&self, lhs: ValueRef, rhs: ValueRef) -> ValueRef {
        self.count_insn("frem");
        unsafe {
            let instr = llvm::LLVMBuildFRem(self.llbuilder, lhs, rhs, noname());
            llvm::LLVMRustSetHasUnsafeAlgebra(instr);
            instr
        }
    }

    pub fn shl(&self, lhs: ValueRef, rhs: ValueRef) -> ValueRef {
        self.count_insn("shl");
        unsafe {
            llvm::LLVMBuildShl(self.llbuilder, lhs, rhs, noname())
        }
    }

    pub fn lshr(&self, lhs: ValueRef, rhs: ValueRef) -> ValueRef {
        self.count_insn("lshr");
        unsafe {
            llvm::LLVMBuildLShr(self.llbuilder, lhs, rhs, noname())
        }
    }

    pub fn ashr(&self, lhs: ValueRef, rhs: ValueRef) -> ValueRef {
        self.count_insn("ashr");
        unsafe {
            llvm::LLVMBuildAShr(self.llbuilder, lhs, rhs, noname())
        }
    }

    pub fn and(&self, lhs: ValueRef, rhs: ValueRef) -> ValueRef {
        self.count_insn("and");
        unsafe {
            llvm::LLVMBuildAnd(self.llbuilder, lhs, rhs, noname())
        }
    }

    pub fn or(&self, lhs: ValueRef, rhs: ValueRef) -> ValueRef {
        self.count_insn("or");
        unsafe {
            llvm::LLVMBuildOr(self.llbuilder, lhs, rhs, noname())
        }
    }

    pub fn xor(&self, lhs: ValueRef, rhs: ValueRef) -> ValueRef {
        self.count_insn("xor");
        unsafe {
            llvm::LLVMBuildXor(self.llbuilder, lhs, rhs, noname())
        }
    }

    pub fn binop(&self, op: Opcode, lhs: ValueRef, rhs: ValueRef)
              -> ValueRef {
        self.count_insn("binop");
        unsafe {
            llvm::LLVMBuildBinOp(self.llbuilder, op, lhs, rhs, noname())
        }
    }

    pub fn neg(&self, v: ValueRef) -> ValueRef {
        self.count_insn("neg");
        unsafe {
            llvm::LLVMBuildNeg(self.llbuilder, v, noname())
        }
    }

    pub fn nswneg(&self, v: ValueRef) -> ValueRef {
        self.count_insn("nswneg");
        unsafe {
            llvm::LLVMBuildNSWNeg(self.llbuilder, v, noname())
        }
    }

    pub fn nuwneg(&self, v: ValueRef) -> ValueRef {
        self.count_insn("nuwneg");
        unsafe {
            llvm::LLVMBuildNUWNeg(self.llbuilder, v, noname())
        }
    }
    pub fn fneg(&self, v: ValueRef) -> ValueRef {
        self.count_insn("fneg");
        unsafe {
            llvm::LLVMBuildFNeg(self.llbuilder, v, noname())
        }
    }

    pub fn not(&self, v: ValueRef) -> ValueRef {
        self.count_insn("not");
        unsafe {
            llvm::LLVMBuildNot(self.llbuilder, v, noname())
        }
    }

    pub fn alloca(&self, ty: Type, name: &str) -> ValueRef {
        let builder = Builder::with_ccx(self.ccx);
        builder.position_at_start(unsafe {
            llvm::LLVMGetFirstBasicBlock(self.llfn())
        });
        builder.dynamic_alloca(ty, name)
    }

    pub fn alloca_ty(&self, ty: Ty<'tcx>, name: &str) -> ValueRef {
        assert!(!ty.has_param_types());
        self.alloca(type_of::type_of(self.ccx, ty), name)
    }

    pub fn dynamic_alloca(&self, ty: Type, name: &str) -> ValueRef {
        self.count_insn("alloca");
        unsafe {
            if name.is_empty() {
                llvm::LLVMBuildAlloca(self.llbuilder, ty.to_ref(), noname())
            } else {
                let name = CString::new(name).unwrap();
                llvm::LLVMBuildAlloca(self.llbuilder, ty.to_ref(),
                                      name.as_ptr())
            }
        }
    }

    pub fn free(&self, ptr: ValueRef) {
        self.count_insn("free");
        unsafe {
            llvm::LLVMBuildFree(self.llbuilder, ptr);
        }
    }

    pub fn load(&self, ptr: ValueRef) -> ValueRef {
        self.count_insn("load");
        unsafe {
            llvm::LLVMBuildLoad(self.llbuilder, ptr, noname())
        }
    }

    pub fn volatile_load(&self, ptr: ValueRef) -> ValueRef {
        self.count_insn("load.volatile");
        unsafe {
            let insn = llvm::LLVMBuildLoad(self.llbuilder, ptr, noname());
            llvm::LLVMSetVolatile(insn, llvm::True);
            insn
        }
    }

    pub fn atomic_load(&self, ptr: ValueRef, order: AtomicOrdering) -> ValueRef {
        self.count_insn("load.atomic");
        unsafe {
            let ty = Type::from_ref(llvm::LLVMTypeOf(ptr));
            let align = llalign_of_pref(self.ccx, ty.element_type());
            llvm::LLVMRustBuildAtomicLoad(self.llbuilder, ptr, noname(), order,
                                          align as c_uint)
        }
    }


    pub fn load_range_assert(&self, ptr: ValueRef, lo: u64,
                             hi: u64, signed: llvm::Bool) -> ValueRef {
        let value = self.load(ptr);

        unsafe {
            let t = llvm::LLVMGetElementType(llvm::LLVMTypeOf(ptr));
            let min = llvm::LLVMConstInt(t, lo, signed);
            let max = llvm::LLVMConstInt(t, hi, signed);

            let v = [min, max];

            llvm::LLVMSetMetadata(value, llvm::MD_range as c_uint,
                                  llvm::LLVMMDNodeInContext(self.ccx.llcx(),
                                                            v.as_ptr(),
                                                            v.len() as c_uint));
        }

        value
    }

    pub fn load_nonnull(&self, ptr: ValueRef) -> ValueRef {
        let value = self.load(ptr);
        unsafe {
            llvm::LLVMSetMetadata(value, llvm::MD_nonnull as c_uint,
                                  llvm::LLVMMDNodeInContext(self.ccx.llcx(), ptr::null(), 0));
        }

        value
    }

    pub fn store(&self, val: ValueRef, ptr: ValueRef, align: Option<u32>) -> ValueRef {
        debug!("Store {:?} -> {:?}", Value(val), Value(ptr));
        assert!(!self.llbuilder.is_null());
        self.count_insn("store");
        let ptr = self.check_store(val, ptr);
        unsafe {
            let store = llvm::LLVMBuildStore(self.llbuilder, val, ptr);
            if let Some(align) = align {
                llvm::LLVMSetAlignment(store, align as c_uint);
            }
            store
        }
    }

    pub fn volatile_store(&self, val: ValueRef, ptr: ValueRef) -> ValueRef {
        debug!("Store {:?} -> {:?}", Value(val), Value(ptr));
        assert!(!self.llbuilder.is_null());
        self.count_insn("store.volatile");
        let ptr = self.check_store(val, ptr);
        unsafe {
            let insn = llvm::LLVMBuildStore(self.llbuilder, val, ptr);
            llvm::LLVMSetVolatile(insn, llvm::True);
            insn
        }
    }

    pub fn atomic_store(&self, val: ValueRef, ptr: ValueRef, order: AtomicOrdering) {
        debug!("Store {:?} -> {:?}", Value(val), Value(ptr));
        self.count_insn("store.atomic");
        let ptr = self.check_store(val, ptr);
        unsafe {
            let ty = Type::from_ref(llvm::LLVMTypeOf(ptr));
            let align = llalign_of_pref(self.ccx, ty.element_type());
            llvm::LLVMRustBuildAtomicStore(self.llbuilder, val, ptr, order, align as c_uint);
        }
    }

    pub fn gep(&self, ptr: ValueRef, indices: &[ValueRef]) -> ValueRef {
        self.count_insn("gep");
        unsafe {
            llvm::LLVMBuildGEP(self.llbuilder, ptr, indices.as_ptr(),
                               indices.len() as c_uint, noname())
        }
    }

    // Simple wrapper around GEP that takes an array of ints and wraps them
    // in C_i32()
    #[inline]
    pub fn gepi(&self, base: ValueRef, ixs: &[usize]) -> ValueRef {
        // Small vector optimization. This should catch 100% of the cases that
        // we care about.
        if ixs.len() < 16 {
            let mut small_vec = [ C_i32(self.ccx, 0); 16 ];
            for (small_vec_e, &ix) in small_vec.iter_mut().zip(ixs) {
                *small_vec_e = C_i32(self.ccx, ix as i32);
            }
            self.inbounds_gep(base, &small_vec[..ixs.len()])
        } else {
            let v = ixs.iter().map(|i| C_i32(self.ccx, *i as i32)).collect::<Vec<ValueRef>>();
            self.count_insn("gepi");
            self.inbounds_gep(base, &v[..])
        }
    }

    pub fn inbounds_gep(&self, ptr: ValueRef, indices: &[ValueRef]) -> ValueRef {
        self.count_insn("inboundsgep");
        unsafe {
            llvm::LLVMBuildInBoundsGEP(
                self.llbuilder, ptr, indices.as_ptr(), indices.len() as c_uint, noname())
        }
    }

    pub fn struct_gep(&self, ptr: ValueRef, idx: usize) -> ValueRef {
        self.count_insn("structgep");
        unsafe {
            llvm::LLVMBuildStructGEP(self.llbuilder, ptr, idx as c_uint, noname())
        }
    }

    pub fn global_string(&self, _str: *const c_char) -> ValueRef {
        self.count_insn("globalstring");
        unsafe {
            llvm::LLVMBuildGlobalString(self.llbuilder, _str, noname())
        }
    }

    pub fn global_string_ptr(&self, _str: *const c_char) -> ValueRef {
        self.count_insn("globalstringptr");
        unsafe {
            llvm::LLVMBuildGlobalStringPtr(self.llbuilder, _str, noname())
        }
    }

    /* Casts */
    pub fn trunc(&self, val: ValueRef, dest_ty: Type) -> ValueRef {
        self.count_insn("trunc");
        unsafe {
            llvm::LLVMBuildTrunc(self.llbuilder, val, dest_ty.to_ref(), noname())
        }
    }

    pub fn zext(&self, val: ValueRef, dest_ty: Type) -> ValueRef {
        self.count_insn("zext");
        unsafe {
            llvm::LLVMBuildZExt(self.llbuilder, val, dest_ty.to_ref(), noname())
        }
    }

    pub fn sext(&self, val: ValueRef, dest_ty: Type) -> ValueRef {
        self.count_insn("sext");
        unsafe {
            llvm::LLVMBuildSExt(self.llbuilder, val, dest_ty.to_ref(), noname())
        }
    }

    pub fn fptoui(&self, val: ValueRef, dest_ty: Type) -> ValueRef {
        self.count_insn("fptoui");
        unsafe {
            llvm::LLVMBuildFPToUI(self.llbuilder, val, dest_ty.to_ref(), noname())
        }
    }

    pub fn fptosi(&self, val: ValueRef, dest_ty: Type) -> ValueRef {
        self.count_insn("fptosi");
        unsafe {
            llvm::LLVMBuildFPToSI(self.llbuilder, val, dest_ty.to_ref(),noname())
        }
    }

    pub fn uitofp(&self, val: ValueRef, dest_ty: Type) -> ValueRef {
        self.count_insn("uitofp");
        unsafe {
            llvm::LLVMBuildUIToFP(self.llbuilder, val, dest_ty.to_ref(), noname())
        }
    }

    pub fn sitofp(&self, val: ValueRef, dest_ty: Type) -> ValueRef {
        self.count_insn("sitofp");
        unsafe {
            llvm::LLVMBuildSIToFP(self.llbuilder, val, dest_ty.to_ref(), noname())
        }
    }

    pub fn fptrunc(&self, val: ValueRef, dest_ty: Type) -> ValueRef {
        self.count_insn("fptrunc");
        unsafe {
            llvm::LLVMBuildFPTrunc(self.llbuilder, val, dest_ty.to_ref(), noname())
        }
    }

    pub fn fpext(&self, val: ValueRef, dest_ty: Type) -> ValueRef {
        self.count_insn("fpext");
        unsafe {
            llvm::LLVMBuildFPExt(self.llbuilder, val, dest_ty.to_ref(), noname())
        }
    }

    pub fn ptrtoint(&self, val: ValueRef, dest_ty: Type) -> ValueRef {
        self.count_insn("ptrtoint");
        unsafe {
            llvm::LLVMBuildPtrToInt(self.llbuilder, val, dest_ty.to_ref(), noname())
        }
    }

    pub fn inttoptr(&self, val: ValueRef, dest_ty: Type) -> ValueRef {
        self.count_insn("inttoptr");
        unsafe {
            llvm::LLVMBuildIntToPtr(self.llbuilder, val, dest_ty.to_ref(), noname())
        }
    }

    pub fn bitcast(&self, val: ValueRef, dest_ty: Type) -> ValueRef {
        self.count_insn("bitcast");
        unsafe {
            llvm::LLVMBuildBitCast(self.llbuilder, val, dest_ty.to_ref(), noname())
        }
    }

    pub fn zext_or_bitcast(&self, val: ValueRef, dest_ty: Type) -> ValueRef {
        self.count_insn("zextorbitcast");
        unsafe {
            llvm::LLVMBuildZExtOrBitCast(self.llbuilder, val, dest_ty.to_ref(), noname())
        }
    }

    pub fn sext_or_bitcast(&self, val: ValueRef, dest_ty: Type) -> ValueRef {
        self.count_insn("sextorbitcast");
        unsafe {
            llvm::LLVMBuildSExtOrBitCast(self.llbuilder, val, dest_ty.to_ref(), noname())
        }
    }

    pub fn trunc_or_bitcast(&self, val: ValueRef, dest_ty: Type) -> ValueRef {
        self.count_insn("truncorbitcast");
        unsafe {
            llvm::LLVMBuildTruncOrBitCast(self.llbuilder, val, dest_ty.to_ref(), noname())
        }
    }

    pub fn cast(&self, op: Opcode, val: ValueRef, dest_ty: Type) -> ValueRef {
        self.count_insn("cast");
        unsafe {
            llvm::LLVMBuildCast(self.llbuilder, op, val, dest_ty.to_ref(), noname())
        }
    }

    pub fn pointercast(&self, val: ValueRef, dest_ty: Type) -> ValueRef {
        self.count_insn("pointercast");
        unsafe {
            llvm::LLVMBuildPointerCast(self.llbuilder, val, dest_ty.to_ref(), noname())
        }
    }

    pub fn intcast(&self, val: ValueRef, dest_ty: Type) -> ValueRef {
        self.count_insn("intcast");
        unsafe {
            llvm::LLVMBuildIntCast(self.llbuilder, val, dest_ty.to_ref(), noname())
        }
    }

    pub fn fpcast(&self, val: ValueRef, dest_ty: Type) -> ValueRef {
        self.count_insn("fpcast");
        unsafe {
            llvm::LLVMBuildFPCast(self.llbuilder, val, dest_ty.to_ref(), noname())
        }
    }


    /* Comparisons */
    pub fn icmp(&self, op: IntPredicate, lhs: ValueRef, rhs: ValueRef) -> ValueRef {
        self.count_insn("icmp");
        unsafe {
            llvm::LLVMBuildICmp(self.llbuilder, op as c_uint, lhs, rhs, noname())
        }
    }

    pub fn fcmp(&self, op: RealPredicate, lhs: ValueRef, rhs: ValueRef) -> ValueRef {
        self.count_insn("fcmp");
        unsafe {
            llvm::LLVMBuildFCmp(self.llbuilder, op as c_uint, lhs, rhs, noname())
        }
    }

    /* Miscellaneous instructions */
    pub fn empty_phi(&self, ty: Type) -> ValueRef {
        self.count_insn("emptyphi");
        unsafe {
            llvm::LLVMBuildPhi(self.llbuilder, ty.to_ref(), noname())
        }
    }

    pub fn phi(&self, ty: Type, vals: &[ValueRef], bbs: &[BasicBlockRef]) -> ValueRef {
        assert_eq!(vals.len(), bbs.len());
        let phi = self.empty_phi(ty);
        self.count_insn("addincoming");
        unsafe {
            llvm::LLVMAddIncoming(phi, vals.as_ptr(),
                                  bbs.as_ptr(),
                                  vals.len() as c_uint);
            phi
        }
    }

    pub fn add_span_comment(&self, sp: Span, text: &str) {
        if self.ccx.sess().asm_comments() {
            let s = format!("{} ({})",
                            text,
                            self.ccx.sess().codemap().span_to_string(sp));
            debug!("{}", &s[..]);
            self.add_comment(&s[..]);
        }
    }

    pub fn add_comment(&self, text: &str) {
        if self.ccx.sess().asm_comments() {
            let sanitized = text.replace("$", "");
            let comment_text = format!("{} {}", "#",
                                       sanitized.replace("\n", "\n\t# "));
            self.count_insn("inlineasm");
            let comment_text = CString::new(comment_text).unwrap();
            let asm = unsafe {
                llvm::LLVMConstInlineAsm(Type::func(&[], &Type::void(self.ccx)).to_ref(),
                                         comment_text.as_ptr(), noname(), False,
                                         False)
            };
            self.call(asm, &[], None);
        }
    }

    pub fn inline_asm_call(&self, asm: *const c_char, cons: *const c_char,
                         inputs: &[ValueRef], output: Type,
                         volatile: bool, alignstack: bool,
                         dia: AsmDialect) -> ValueRef {
        self.count_insn("inlineasm");

        let volatile = if volatile { llvm::True }
                       else        { llvm::False };
        let alignstack = if alignstack { llvm::True }
                         else          { llvm::False };

        let argtys = inputs.iter().map(|v| {
            debug!("Asm Input Type: {:?}", Value(*v));
            val_ty(*v)
        }).collect::<Vec<_>>();

        debug!("Asm Output Type: {:?}", output);
        let fty = Type::func(&argtys[..], &output);
        unsafe {
            let v = llvm::LLVMRustInlineAsm(
                fty.to_ref(), asm, cons, volatile, alignstack, dia);
            self.call(v, inputs, None)
        }
    }

    pub fn call(&self, llfn: ValueRef, args: &[ValueRef],
                bundle: Option<&OperandBundleDef>) -> ValueRef {
        self.count_insn("call");

        debug!("Call {:?} with args ({})",
               Value(llfn),
               args.iter()
                   .map(|&v| format!("{:?}", Value(v)))
                   .collect::<Vec<String>>()
                   .join(", "));

        let args = self.check_call("call", llfn, args);
        let bundle = bundle.as_ref().map(|b| b.raw()).unwrap_or(ptr::null_mut());

        unsafe {
            llvm::LLVMRustBuildCall(self.llbuilder, llfn, args.as_ptr(),
                                    args.len() as c_uint, bundle, noname())
        }
    }

    pub fn select(&self, cond: ValueRef, then_val: ValueRef, else_val: ValueRef) -> ValueRef {
        self.count_insn("select");
        unsafe {
            llvm::LLVMBuildSelect(self.llbuilder, cond, then_val, else_val, noname())
        }
    }

    pub fn va_arg(&self, list: ValueRef, ty: Type) -> ValueRef {
        self.count_insn("vaarg");
        unsafe {
            llvm::LLVMBuildVAArg(self.llbuilder, list, ty.to_ref(), noname())
        }
    }

    pub fn extract_element(&self, vec: ValueRef, idx: ValueRef) -> ValueRef {
        self.count_insn("extractelement");
        unsafe {
            llvm::LLVMBuildExtractElement(self.llbuilder, vec, idx, noname())
        }
    }

    pub fn insert_element(&self, vec: ValueRef, elt: ValueRef, idx: ValueRef) -> ValueRef {
        self.count_insn("insertelement");
        unsafe {
            llvm::LLVMBuildInsertElement(self.llbuilder, vec, elt, idx, noname())
        }
    }

    pub fn shuffle_vector(&self, v1: ValueRef, v2: ValueRef, mask: ValueRef) -> ValueRef {
        self.count_insn("shufflevector");
        unsafe {
            llvm::LLVMBuildShuffleVector(self.llbuilder, v1, v2, mask, noname())
        }
    }

    pub fn vector_splat(&self, num_elts: usize, elt: ValueRef) -> ValueRef {
        unsafe {
            let elt_ty = val_ty(elt);
            let undef = llvm::LLVMGetUndef(Type::vector(&elt_ty, num_elts as u64).to_ref());
            let vec = self.insert_element(undef, elt, C_i32(self.ccx, 0));
            let vec_i32_ty = Type::vector(&Type::i32(self.ccx), num_elts as u64);
            self.shuffle_vector(vec, undef, C_null(vec_i32_ty))
        }
    }

    pub fn extract_value(&self, agg_val: ValueRef, idx: usize) -> ValueRef {
        self.count_insn("extractvalue");
        unsafe {
            llvm::LLVMBuildExtractValue(self.llbuilder, agg_val, idx as c_uint, noname())
        }
    }

    pub fn insert_value(&self, agg_val: ValueRef, elt: ValueRef,
                       idx: usize) -> ValueRef {
        self.count_insn("insertvalue");
        unsafe {
            llvm::LLVMBuildInsertValue(self.llbuilder, agg_val, elt, idx as c_uint,
                                       noname())
        }
    }

    pub fn is_null(&self, val: ValueRef) -> ValueRef {
        self.count_insn("isnull");
        unsafe {
            llvm::LLVMBuildIsNull(self.llbuilder, val, noname())
        }
    }

    pub fn is_not_null(&self, val: ValueRef) -> ValueRef {
        self.count_insn("isnotnull");
        unsafe {
            llvm::LLVMBuildIsNotNull(self.llbuilder, val, noname())
        }
    }

    pub fn ptrdiff(&self, lhs: ValueRef, rhs: ValueRef) -> ValueRef {
        self.count_insn("ptrdiff");
        unsafe {
            llvm::LLVMBuildPtrDiff(self.llbuilder, lhs, rhs, noname())
        }
    }

    pub fn trap(&self) {
        unsafe {
            let bb: BasicBlockRef = llvm::LLVMGetInsertBlock(self.llbuilder);
            let fn_: ValueRef = llvm::LLVMGetBasicBlockParent(bb);
            let m: ModuleRef = llvm::LLVMGetGlobalParent(fn_);
            let p = "llvm.trap\0".as_ptr();
            let t: ValueRef = llvm::LLVMGetNamedFunction(m, p as *const _);
            assert!((t as isize != 0));
            let args: &[ValueRef] = &[];
            self.count_insn("trap");
            llvm::LLVMRustBuildCall(self.llbuilder, t,
                                    args.as_ptr(), args.len() as c_uint,
                                    ptr::null_mut(),
                                    noname());
        }
    }

    pub fn landing_pad(&self, ty: Type, pers_fn: ValueRef,
                       num_clauses: usize,
                       llfn: ValueRef) -> ValueRef {
        self.count_insn("landingpad");
        unsafe {
            llvm::LLVMRustBuildLandingPad(self.llbuilder, ty.to_ref(), pers_fn,
                                          num_clauses as c_uint, noname(), llfn)
        }
    }

    pub fn add_clause(&self, landing_pad: ValueRef, clause: ValueRef) {
        unsafe {
            llvm::LLVMAddClause(landing_pad, clause);
        }
    }

    pub fn set_cleanup(&self, landing_pad: ValueRef) {
        self.count_insn("setcleanup");
        unsafe {
            llvm::LLVMSetCleanup(landing_pad, llvm::True);
        }
    }

    pub fn resume(&self, exn: ValueRef) -> ValueRef {
        self.count_insn("resume");
        unsafe {
            llvm::LLVMBuildResume(self.llbuilder, exn)
        }
    }

    pub fn cleanup_pad(&self,
                       parent: Option<ValueRef>,
                       args: &[ValueRef]) -> ValueRef {
        self.count_insn("cleanuppad");
        let parent = parent.unwrap_or(ptr::null_mut());
        let name = CString::new("cleanuppad").unwrap();
        let ret = unsafe {
            llvm::LLVMRustBuildCleanupPad(self.llbuilder,
                                          parent,
                                          args.len() as c_uint,
                                          args.as_ptr(),
                                          name.as_ptr())
        };
        assert!(!ret.is_null(), "LLVM does not have support for cleanuppad");
        return ret
    }

    pub fn cleanup_ret(&self, cleanup: ValueRef,
                       unwind: Option<BasicBlockRef>) -> ValueRef {
        self.count_insn("cleanupret");
        let unwind = unwind.unwrap_or(ptr::null_mut());
        let ret = unsafe {
            llvm::LLVMRustBuildCleanupRet(self.llbuilder, cleanup, unwind)
        };
        assert!(!ret.is_null(), "LLVM does not have support for cleanupret");
        return ret
    }

    pub fn catch_pad(&self,
                     parent: ValueRef,
                     args: &[ValueRef]) -> ValueRef {
        self.count_insn("catchpad");
        let name = CString::new("catchpad").unwrap();
        let ret = unsafe {
            llvm::LLVMRustBuildCatchPad(self.llbuilder, parent,
                                        args.len() as c_uint, args.as_ptr(),
                                        name.as_ptr())
        };
        assert!(!ret.is_null(), "LLVM does not have support for catchpad");
        return ret
    }

    pub fn catch_ret(&self, pad: ValueRef, unwind: BasicBlockRef) -> ValueRef {
        self.count_insn("catchret");
        let ret = unsafe {
            llvm::LLVMRustBuildCatchRet(self.llbuilder, pad, unwind)
        };
        assert!(!ret.is_null(), "LLVM does not have support for catchret");
        return ret
    }

    pub fn catch_switch(&self,
                        parent: Option<ValueRef>,
                        unwind: Option<BasicBlockRef>,
                        num_handlers: usize) -> ValueRef {
        self.count_insn("catchswitch");
        let parent = parent.unwrap_or(ptr::null_mut());
        let unwind = unwind.unwrap_or(ptr::null_mut());
        let name = CString::new("catchswitch").unwrap();
        let ret = unsafe {
            llvm::LLVMRustBuildCatchSwitch(self.llbuilder, parent, unwind,
                                           num_handlers as c_uint,
                                           name.as_ptr())
        };
        assert!(!ret.is_null(), "LLVM does not have support for catchswitch");
        return ret
    }

    pub fn add_handler(&self, catch_switch: ValueRef, handler: BasicBlockRef) {
        unsafe {
            llvm::LLVMRustAddHandler(catch_switch, handler);
        }
    }

    pub fn set_personality_fn(&self, personality: ValueRef) {
        unsafe {
            llvm::LLVMRustSetPersonalityFn(self.llbuilder, personality);
        }
    }

    // Atomic Operations
    pub fn atomic_cmpxchg(&self, dst: ValueRef,
                         cmp: ValueRef, src: ValueRef,
                         order: AtomicOrdering,
                         failure_order: AtomicOrdering,
                         weak: llvm::Bool) -> ValueRef {
        unsafe {
            llvm::LLVMRustBuildAtomicCmpXchg(self.llbuilder, dst, cmp, src,
                                         order, failure_order, weak)
        }
    }
    pub fn atomic_rmw(&self, op: AtomicRmwBinOp,
                     dst: ValueRef, src: ValueRef,
                     order: AtomicOrdering) -> ValueRef {
        unsafe {
            llvm::LLVMBuildAtomicRMW(self.llbuilder, op, dst, src, order, False)
        }
    }

    pub fn atomic_fence(&self, order: AtomicOrdering, scope: SynchronizationScope) {
        unsafe {
            llvm::LLVMRustBuildAtomicFence(self.llbuilder, order, scope);
        }
    }

    pub fn add_case(&self, s: ValueRef, on_val: ValueRef, dest: BasicBlockRef) {
        unsafe {
            if llvm::LLVMIsUndef(s) == llvm::True { return; }
            llvm::LLVMAddCase(s, on_val, dest)
        }
    }

    pub fn add_incoming_to_phi(&self, phi: ValueRef, val: ValueRef, bb: BasicBlockRef) {
        unsafe {
            if llvm::LLVMIsUndef(phi) == llvm::True { return; }
            llvm::LLVMAddIncoming(phi, &val, &bb, 1 as c_uint);
        }
    }

    /// Returns the ptr value that should be used for storing `val`.
    fn check_store<'b>(&self,
                       val: ValueRef,
                       ptr: ValueRef) -> ValueRef {
        let dest_ptr_ty = val_ty(ptr);
        let stored_ty = val_ty(val);
        let stored_ptr_ty = stored_ty.ptr_to();

        assert_eq!(dest_ptr_ty.kind(), llvm::TypeKind::Pointer);

        if dest_ptr_ty == stored_ptr_ty {
            ptr
        } else {
            debug!("Type mismatch in store. \
                    Expected {:?}, got {:?}; inserting bitcast",
                   dest_ptr_ty, stored_ptr_ty);
            self.bitcast(ptr, stored_ptr_ty)
        }
    }

    /// Returns the args that should be used for a call to `llfn`.
    fn check_call<'b>(&self,
                      typ: &str,
                      llfn: ValueRef,
                      args: &'b [ValueRef]) -> Cow<'b, [ValueRef]> {
        let mut fn_ty = val_ty(llfn);
        // Strip off pointers
        while fn_ty.kind() == llvm::TypeKind::Pointer {
            fn_ty = fn_ty.element_type();
        }

        assert!(fn_ty.kind() == llvm::TypeKind::Function,
                "builder::{} not passed a function", typ);

        let param_tys = fn_ty.func_params();

        let all_args_match = param_tys.iter()
            .zip(args.iter().map(|&v| val_ty(v)))
            .all(|(expected_ty, actual_ty)| *expected_ty == actual_ty);

        if all_args_match {
            return Cow::Borrowed(args);
        }

        let casted_args: Vec<_> = param_tys.into_iter()
            .zip(args.iter())
            .enumerate()
            .map(|(i, (expected_ty, &actual_val))| {
                let actual_ty = val_ty(actual_val);
                if expected_ty != actual_ty {
                    debug!("Type mismatch in function call of {:?}. \
                            Expected {:?} for param {}, got {:?}; injecting bitcast",
                           Value(llfn),
                           expected_ty, i, actual_ty);
                    self.bitcast(actual_val, expected_ty)
                } else {
                    actual_val
                }
            })
            .collect();

        return Cow::Owned(casted_args);
    }
}
