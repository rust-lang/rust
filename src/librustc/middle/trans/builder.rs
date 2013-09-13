// Copyright 2013 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use lib;
use lib::llvm::llvm;
use lib::llvm::{CallConv, AtomicBinOp, AtomicOrdering, AsmDialect};
use lib::llvm::{Opcode, IntPredicate, RealPredicate, False};
use lib::llvm::{ValueRef, BasicBlockRef, BuilderRef, ModuleRef};
use middle::trans::base;
use middle::trans::common::*;
use middle::trans::machine::llalign_of_min;
use middle::trans::type_::Type;
use std::cast;
use std::hashmap::HashMap;
use std::libc::{c_uint, c_ulonglong, c_char};
use std::vec;
use syntax::codemap::Span;
use std::ptr::is_not_null;

pub struct Builder {
    llbuilder: BuilderRef,
    ccx: @mut CrateContext,
}

// This is a really awful way to get a zero-length c-string, but better (and a
// lot more efficient) than doing str::as_c_str("", ...) every time.
pub fn noname() -> *c_char {
    unsafe {
        static cnull: uint = 0u;
        cast::transmute(&cnull)
    }
}

impl Builder {
    pub fn new(ccx: @mut CrateContext) -> Builder {
        Builder {
            llbuilder: ccx.builder.B,
            ccx: ccx,
        }
    }

    pub fn count_insn(&self, category: &str) {
        if self.ccx.sess.trans_stats() {
            self.ccx.stats.n_llvm_insns += 1;
        }
        if self.ccx.sess.count_llvm_insns() {
            do base::with_insn_ctxt |v| {
                let h = &mut self.ccx.stats.llvm_insns;

                // Build version of path with cycles removed.

                // Pass 1: scan table mapping str -> rightmost pos.
                let mut mm = HashMap::new();
                let len = v.len();
                let mut i = 0u;
                while i < len {
                    mm.insert(v[i], i);
                    i += 1u;
                }

                // Pass 2: concat strings for each elt, skipping
                // forwards over any cycles by advancing to rightmost
                // occurrence of each element in path.
                let mut s = ~".";
                i = 0u;
                while i < len {
                    i = *mm.get(&v[i]);
                    s.push_char('/');
                    s.push_str(v[i]);
                    i += 1u;
                }

                s.push_char('/');
                s.push_str(category);

                let n = match h.find(&s) {
                    Some(&n) => n,
                    _ => 0u
                };
                h.insert(s, n+1u);
            }
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
                                        vec::raw::to_ptr(ret_vals),
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

    pub fn switch(&self, v: ValueRef, else_llbb: BasicBlockRef, num_cases: uint) -> ValueRef {
        unsafe {
            llvm::LLVMBuildSwitch(self.llbuilder, v, else_llbb, num_cases as c_uint)
        }
    }

    pub fn indirect_br(&self, addr: ValueRef, num_dests: uint) {
        self.count_insn("indirectbr");
        unsafe {
            llvm::LLVMBuildIndirectBr(self.llbuilder, addr, num_dests as c_uint);
        }
    }

    pub fn invoke(&self,
                  llfn: ValueRef,
                  args: &[ValueRef],
                  then: BasicBlockRef,
                  catch: BasicBlockRef)
                  -> ValueRef {
        self.count_insn("invoke");
        unsafe {
            llvm::LLVMBuildInvoke(self.llbuilder,
                                  llfn,
                                  vec::raw::to_ptr(args),
                                  args.len() as c_uint,
                                  then,
                                  catch,
                                  noname())
        }
    }

    pub fn fast_invoke(&self,
                       llfn: ValueRef,
                       args: &[ValueRef],
                       then: BasicBlockRef,
                       catch: BasicBlockRef) {
        self.count_insn("fastinvoke");
        let v = self.invoke(llfn, args, then, catch);
        lib::llvm::SetInstructionCallConv(v, lib::llvm::FastCallConv);
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

    pub fn neg(&self, V: ValueRef) -> ValueRef {
        self.count_insn("neg");
        unsafe {
            llvm::LLVMBuildNeg(self.llbuilder, V, noname())
        }
    }

    pub fn nswneg(&self, V: ValueRef) -> ValueRef {
        self.count_insn("nswneg");
        unsafe {
            llvm::LLVMBuildNSWNeg(self.llbuilder, V, noname())
        }
    }

    pub fn nuwneg(&self, V: ValueRef) -> ValueRef {
        self.count_insn("nuwneg");
        unsafe {
            llvm::LLVMBuildNUWNeg(self.llbuilder, V, noname())
        }
    }
    pub fn fneg(&self, V: ValueRef) -> ValueRef {
        self.count_insn("fneg");
        unsafe {
            llvm::LLVMBuildFNeg(self.llbuilder, V, noname())
        }
    }

    pub fn not(&self, V: ValueRef) -> ValueRef {
        self.count_insn("not");
        unsafe {
            llvm::LLVMBuildNot(self.llbuilder, V, noname())
        }
    }

    /* Memory */
    pub fn malloc(&self, ty: Type) -> ValueRef {
        self.count_insn("malloc");
        unsafe {
            llvm::LLVMBuildMalloc(self.llbuilder, ty.to_ref(), noname())
        }
    }

    pub fn array_malloc(&self, ty: Type, val: ValueRef) -> ValueRef {
        self.count_insn("arraymalloc");
        unsafe {
            llvm::LLVMBuildArrayMalloc(self.llbuilder, ty.to_ref(), val, noname())
        }
    }

    pub fn alloca(&self, ty: Type, name: &str) -> ValueRef {
        self.count_insn("alloca");
        unsafe {
            if name.is_empty() {
                llvm::LLVMBuildAlloca(self.llbuilder, ty.to_ref(), noname())
            } else {
                do name.with_c_str |c| {
                    llvm::LLVMBuildAlloca(self.llbuilder, ty.to_ref(), c)
                }
            }
        }
    }

    pub fn array_alloca(&self, ty: Type, val: ValueRef) -> ValueRef {
        self.count_insn("arrayalloca");
        unsafe {
            llvm::LLVMBuildArrayAlloca(self.llbuilder, ty.to_ref(), val, noname())
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

    pub fn atomic_load(&self, ptr: ValueRef, order: AtomicOrdering) -> ValueRef {
        self.count_insn("load.atomic");
        unsafe {
            let align = llalign_of_min(self.ccx, self.ccx.int_type);
            llvm::LLVMBuildAtomicLoad(self.llbuilder, ptr, noname(), order, align as c_uint)
        }
    }


    pub fn load_range_assert(&self, ptr: ValueRef, lo: c_ulonglong,
                           hi: c_ulonglong, signed: lib::llvm::Bool) -> ValueRef {
        let value = self.load(ptr);

        unsafe {
            let t = llvm::LLVMGetElementType(llvm::LLVMTypeOf(ptr));
            let min = llvm::LLVMConstInt(t, lo, signed);
            let max = llvm::LLVMConstInt(t, hi, signed);

            do [min, max].as_imm_buf |ptr, len| {
                llvm::LLVMSetMetadata(value, lib::llvm::MD_range as c_uint,
                                      llvm::LLVMMDNodeInContext(self.ccx.llcx,
                                                                ptr, len as c_uint));
            }
        }

        value
    }

    pub fn store(&self, val: ValueRef, ptr: ValueRef) {
        debug!("Store %s -> %s",
               self.ccx.tn.val_to_str(val),
               self.ccx.tn.val_to_str(ptr));
        assert!(is_not_null(self.llbuilder));
        self.count_insn("store");
        unsafe {
            llvm::LLVMBuildStore(self.llbuilder, val, ptr);
        }
    }

    pub fn atomic_store(&self, val: ValueRef, ptr: ValueRef, order: AtomicOrdering) {
        debug!("Store %s -> %s",
               self.ccx.tn.val_to_str(val),
               self.ccx.tn.val_to_str(ptr));
        self.count_insn("store.atomic");
        let align = llalign_of_min(self.ccx, self.ccx.int_type);
        unsafe {
            llvm::LLVMBuildAtomicStore(self.llbuilder, val, ptr, order, align as c_uint);
        }
    }

    pub fn gep(&self, ptr: ValueRef, indices: &[ValueRef]) -> ValueRef {
        self.count_insn("gep");
        unsafe {
            llvm::LLVMBuildGEP(self.llbuilder, ptr, vec::raw::to_ptr(indices),
                               indices.len() as c_uint, noname())
        }
    }

    // Simple wrapper around GEP that takes an array of ints and wraps them
    // in C_i32()
    #[inline]
    pub fn gepi(&self, base: ValueRef, ixs: &[uint]) -> ValueRef {
        // Small vector optimization. This should catch 100% of the cases that
        // we care about.
        if ixs.len() < 16 {
            let mut small_vec = [ C_i32(0), ..16 ];
            for (small_vec_e, &ix) in small_vec.mut_iter().zip(ixs.iter()) {
                *small_vec_e = C_i32(ix as i32);
            }
            self.inbounds_gep(base, small_vec.slice(0, ixs.len()))
        } else {
            let v = do ixs.iter().map |i| { C_i32(*i as i32) }.collect::<~[ValueRef]>();
            self.count_insn("gepi");
            self.inbounds_gep(base, v)
        }
    }

    pub fn inbounds_gep(&self, ptr: ValueRef, indices: &[ValueRef]) -> ValueRef {
        self.count_insn("inboundsgep");
        unsafe {
            llvm::LLVMBuildInBoundsGEP(
                self.llbuilder, ptr, vec::raw::to_ptr(indices), indices.len() as c_uint, noname())
        }
    }

    pub fn struct_gep(&self, ptr: ValueRef, idx: uint) -> ValueRef {
        self.count_insn("structgep");
        unsafe {
            llvm::LLVMBuildStructGEP(self.llbuilder, ptr, idx as c_uint, noname())
        }
    }

    pub fn global_string(&self, _Str: *c_char) -> ValueRef {
        self.count_insn("globalstring");
        unsafe {
            llvm::LLVMBuildGlobalString(self.llbuilder, _Str, noname())
        }
    }

    pub fn global_string_ptr(&self, _Str: *c_char) -> ValueRef {
        self.count_insn("globalstringptr");
        unsafe {
            llvm::LLVMBuildGlobalStringPtr(self.llbuilder, _Str, noname())
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
            llvm::LLVMAddIncoming(phi, vec::raw::to_ptr(vals),
                                  vec::raw::to_ptr(bbs),
                                  vals.len() as c_uint);
            phi
        }
    }

    pub fn add_span_comment(&self, sp: Span, text: &str) {
        if self.ccx.sess.asm_comments() {
            let s = fmt!("%s (%s)", text, self.ccx.sess.codemap.span_to_str(sp));
            debug!("%s", s);
            self.add_comment(s);
        }
    }

    pub fn add_comment(&self, text: &str) {
        if self.ccx.sess.asm_comments() {
            let sanitized = text.replace("$", "");
            let comment_text = fmt!("# %s", sanitized.replace("\n", "\n\t# "));
            self.count_insn("inlineasm");
            let asm = do comment_text.with_c_str |c| {
                unsafe {
                    llvm::LLVMConstInlineAsm(Type::func([], &Type::void()).to_ref(),
                                             c, noname(), False, False)
                }
            };
            self.call(asm, [], []);
        }
    }

    pub fn inline_asm_call(&self, asm: *c_char, cons: *c_char,
                         inputs: &[ValueRef], output: Type,
                         volatile: bool, alignstack: bool,
                         dia: AsmDialect) -> ValueRef {
        self.count_insn("inlineasm");

        let volatile = if volatile { lib::llvm::True }
                       else        { lib::llvm::False };
        let alignstack = if alignstack { lib::llvm::True }
                         else          { lib::llvm::False };

        let argtys = do inputs.map |v| {
            debug!("Asm Input Type: %?", self.ccx.tn.val_to_str(*v));
            val_ty(*v)
        };

        debug!("Asm Output Type: %?", self.ccx.tn.type_to_str(output));
        let fty = Type::func(argtys, &output);
        unsafe {
            let v = llvm::LLVMInlineAsm(
                fty.to_ref(), asm, cons, volatile, alignstack, dia as c_uint);
            self.call(v, inputs, [])
        }
    }

    pub fn call(&self, llfn: ValueRef, args: &[ValueRef],
                attributes: &[(uint, lib::llvm::Attribute)]) -> ValueRef {
        self.count_insn("call");
        unsafe {
            let v = llvm::LLVMBuildCall(self.llbuilder, llfn, vec::raw::to_ptr(args),
                                        args.len() as c_uint, noname());
            for &(idx, attr) in attributes.iter() {
                llvm::LLVMAddInstrAttribute(v, idx as c_uint, attr as c_uint);
            }
            v
        }
    }

    pub fn call_with_conv(&self, llfn: ValueRef, args: &[ValueRef],
                          conv: CallConv, attributes: &[(uint, lib::llvm::Attribute)]) -> ValueRef {
        self.count_insn("callwithconv");
        let v = self.call(llfn, args, attributes);
        lib::llvm::SetInstructionCallConv(v, conv);
        v
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

    pub fn vector_splat(&self, num_elts: uint, elt: ValueRef) -> ValueRef {
        unsafe {
            let elt_ty = val_ty(elt);
            let Undef = llvm::LLVMGetUndef(Type::vector(&elt_ty, num_elts as u64).to_ref());
            let vec = self.insert_element(Undef, elt, C_i32(0));
            self.shuffle_vector(vec, Undef, C_null(Type::vector(&Type::i32(), num_elts as u64)))
        }
    }

    pub fn extract_value(&self, agg_val: ValueRef, idx: uint) -> ValueRef {
        self.count_insn("extractvalue");
        unsafe {
            llvm::LLVMBuildExtractValue(self.llbuilder, agg_val, idx as c_uint, noname())
        }
    }

    pub fn insert_value(&self, agg_val: ValueRef, elt: ValueRef,
                       idx: uint) -> ValueRef {
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
            let BB: BasicBlockRef = llvm::LLVMGetInsertBlock(self.llbuilder);
            let FN: ValueRef = llvm::LLVMGetBasicBlockParent(BB);
            let M: ModuleRef = llvm::LLVMGetGlobalParent(FN);
            let T: ValueRef = do "llvm.trap".with_c_str |buf| {
                llvm::LLVMGetNamedFunction(M, buf)
            };
            assert!((T as int != 0));
            let args: &[ValueRef] = [];
            self.count_insn("trap");
            llvm::LLVMBuildCall(
                self.llbuilder, T, vec::raw::to_ptr(args), args.len() as c_uint, noname());
        }
    }

    pub fn landing_pad(&self, ty: Type, pers_fn: ValueRef, num_clauses: uint) -> ValueRef {
        self.count_insn("landingpad");
        unsafe {
            llvm::LLVMBuildLandingPad(
                self.llbuilder, ty.to_ref(), pers_fn, num_clauses as c_uint, noname())
        }
    }

    pub fn set_cleanup(&self, landing_pad: ValueRef) {
        self.count_insn("setcleanup");
        unsafe {
            llvm::LLVMSetCleanup(landing_pad, lib::llvm::True);
        }
    }

    pub fn resume(&self, exn: ValueRef) -> ValueRef {
        self.count_insn("resume");
        unsafe {
            llvm::LLVMBuildResume(self.llbuilder, exn)
        }
    }

    // Atomic Operations
    pub fn atomic_cmpxchg(&self, dst: ValueRef,
                         cmp: ValueRef, src: ValueRef,
                         order: AtomicOrdering) -> ValueRef {
        unsafe {
            llvm::LLVMBuildAtomicCmpXchg(self.llbuilder, dst, cmp, src, order)
        }
    }
    pub fn atomic_rmw(&self, op: AtomicBinOp,
                     dst: ValueRef, src: ValueRef,
                     order: AtomicOrdering) -> ValueRef {
        unsafe {
            llvm::LLVMBuildAtomicRMW(self.llbuilder, op, dst, src, order, False)
        }
    }

    pub fn atomic_fence(&self, order: AtomicOrdering) {
        unsafe {
            llvm::LLVMBuildAtomicFence(self.llbuilder, order);
        }
    }
}
