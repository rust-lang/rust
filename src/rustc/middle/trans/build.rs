use std::map::{HashMap, str_hash};
use libc::{c_uint, c_int};
use lib::llvm::llvm;
use syntax::codemap;
use codemap::span;
use lib::llvm::{ValueRef, TypeRef, BasicBlockRef, BuilderRef, ModuleRef};
use lib::llvm::{Opcode, IntPredicate, RealPredicate, True, False,
        CallConv, TypeKind, AtomicBinOp, AtomicOrdering};
use driver::session::session;
use common::*;

fn B(cx: block) -> BuilderRef {
    let b = cx.fcx.ccx.builder.B;
    llvm::LLVMPositionBuilderAtEnd(b, cx.llbb);
    return b;
}

fn count_insn(cx: block, category: &str) {
    if cx.ccx().sess.count_llvm_insns() {

        let h = cx.ccx().stats.llvm_insns;
        let v = cx.ccx().stats.llvm_insn_ctxt;

        // Build version of path with cycles removed.

        // Pass 1: scan table mapping str -> rightmost pos.
        let mm = str_hash();
        let len = vec::len(*v);
        let mut i = 0u;
        while i < len {
            mm.insert(copy v[i], i);
            i += 1u;
        }


        // Pass 2: concat strings for each elt, skipping
        // forwards over any cycles by advancing to rightmost
        // occurrence of each element in path.
        let mut s = ~".";
        i = 0u;
        while i < len {
            let e = v[i];
            i = mm.get(e);
            s += ~"/";
            s += e;
            i += 1u;
        }

        s += ~"/";
        s += category;

        let n = match h.find(s) {
          Some(n) => n,
          _ => 0u
        };
        h.insert(s, n+1u);
    }
}


// The difference between a block being unreachable and being terminated is
// somewhat obscure, and has to do with error checking. When a block is
// terminated, we're saying that trying to add any further statements in the
// block is an error. On the other hand, if something is unreachable, that
// means that the block was terminated in some way that we don't want to check
// for (fail/break/return statements, call to diverging functions, etc), and
// further instructions to the block should simply be ignored.

fn RetVoid(cx: block) {
    if cx.unreachable { return; }
    assert (!cx.terminated);
    cx.terminated = true;
    count_insn(cx, "retvoid");
    llvm::LLVMBuildRetVoid(B(cx));
}

fn Ret(cx: block, V: ValueRef) {
    if cx.unreachable { return; }
    assert (!cx.terminated);
    cx.terminated = true;
    count_insn(cx, "ret");
    llvm::LLVMBuildRet(B(cx), V);
}

fn AggregateRet(cx: block, RetVals: ~[ValueRef]) {
    if cx.unreachable { return; }
    assert (!cx.terminated);
    cx.terminated = true;
    unsafe {
        llvm::LLVMBuildAggregateRet(B(cx), vec::unsafe::to_ptr(RetVals),
                                    RetVals.len() as c_uint);
    }
}

fn Br(cx: block, Dest: BasicBlockRef) {
    if cx.unreachable { return; }
    assert (!cx.terminated);
    cx.terminated = true;
    count_insn(cx, "br");
    llvm::LLVMBuildBr(B(cx), Dest);
}

fn CondBr(cx: block, If: ValueRef, Then: BasicBlockRef,
          Else: BasicBlockRef) {
    if cx.unreachable { return; }
    assert (!cx.terminated);
    cx.terminated = true;
    count_insn(cx, "condbr");
    llvm::LLVMBuildCondBr(B(cx), If, Then, Else);
}

fn Switch(cx: block, V: ValueRef, Else: BasicBlockRef, NumCases: uint)
    -> ValueRef {
    if cx.unreachable { return _Undef(V); }
    assert !cx.terminated;
    cx.terminated = true;
    return llvm::LLVMBuildSwitch(B(cx), V, Else, NumCases as c_uint);
}

fn AddCase(S: ValueRef, OnVal: ValueRef, Dest: BasicBlockRef) {
    if llvm::LLVMIsUndef(S) == lib::llvm::True { return; }
    llvm::LLVMAddCase(S, OnVal, Dest);
}

fn IndirectBr(cx: block, Addr: ValueRef, NumDests: uint) {
    if cx.unreachable { return; }
    assert (!cx.terminated);
    cx.terminated = true;
    count_insn(cx, "indirectbr");
    llvm::LLVMBuildIndirectBr(B(cx), Addr, NumDests as c_uint);
}

// This is a really awful way to get a zero-length c-string, but better (and a
// lot more efficient) than doing str::as_c_str("", ...) every time.
fn noname() -> *libc::c_char unsafe {
    const cnull: uint = 0u;
    return unsafe::reinterpret_cast(&ptr::addr_of(cnull));
}

fn Invoke(cx: block, Fn: ValueRef, Args: ~[ValueRef],
          Then: BasicBlockRef, Catch: BasicBlockRef) {
    if cx.unreachable { return; }
    assert (!cx.terminated);
    cx.terminated = true;
    debug!("Invoke(%s with arguments (%s))",
           val_str(cx.ccx().tn, Fn),
           str::connect(vec::map(Args, |a| val_str(cx.ccx().tn, a)),
                        ~", "));
    unsafe {
        count_insn(cx, "invoke");
        llvm::LLVMBuildInvoke(B(cx), Fn, vec::unsafe::to_ptr(Args),
                              Args.len() as c_uint, Then, Catch,
                              noname());
    }
}

fn FastInvoke(cx: block, Fn: ValueRef, Args: ~[ValueRef],
              Then: BasicBlockRef, Catch: BasicBlockRef) {
    if cx.unreachable { return; }
    assert (!cx.terminated);
    cx.terminated = true;
    unsafe {
        count_insn(cx, "fastinvoke");
        let v = llvm::LLVMBuildInvoke(B(cx), Fn, vec::unsafe::to_ptr(Args),
                                      Args.len() as c_uint,
                                      Then, Catch, noname());
        lib::llvm::SetInstructionCallConv(v, lib::llvm::FastCallConv);
    }
}

fn Unreachable(cx: block) {
    if cx.unreachable { return; }
    cx.unreachable = true;
    if !cx.terminated {
        count_insn(cx, "unreachable");
        llvm::LLVMBuildUnreachable(B(cx));
    }
}

fn _Undef(val: ValueRef) -> ValueRef {
    return llvm::LLVMGetUndef(val_ty(val));
}

/* Arithmetic */
fn Add(cx: block, LHS: ValueRef, RHS: ValueRef) -> ValueRef {
    if cx.unreachable { return _Undef(LHS); }
    count_insn(cx, "add");
    return llvm::LLVMBuildAdd(B(cx), LHS, RHS, noname());
}

fn NSWAdd(cx: block, LHS: ValueRef, RHS: ValueRef) -> ValueRef {
    if cx.unreachable { return _Undef(LHS); }
    count_insn(cx, "nswadd");
    return llvm::LLVMBuildNSWAdd(B(cx), LHS, RHS, noname());
}

fn NUWAdd(cx: block, LHS: ValueRef, RHS: ValueRef) -> ValueRef {
    if cx.unreachable { return _Undef(LHS); }
    count_insn(cx, "nuwadd");
    return llvm::LLVMBuildNUWAdd(B(cx), LHS, RHS, noname());
}

fn FAdd(cx: block, LHS: ValueRef, RHS: ValueRef) -> ValueRef {
    if cx.unreachable { return _Undef(LHS); }
    count_insn(cx, "fadd");
    return llvm::LLVMBuildFAdd(B(cx), LHS, RHS, noname());
}

fn Sub(cx: block, LHS: ValueRef, RHS: ValueRef) -> ValueRef {
    if cx.unreachable { return _Undef(LHS); }
    count_insn(cx, "sub");
    return llvm::LLVMBuildSub(B(cx), LHS, RHS, noname());
}

fn NSWSub(cx: block, LHS: ValueRef, RHS: ValueRef) -> ValueRef {
    if cx.unreachable { return _Undef(LHS); }
    count_insn(cx, "nwsub");
    return llvm::LLVMBuildNSWSub(B(cx), LHS, RHS, noname());
}

fn NUWSub(cx: block, LHS: ValueRef, RHS: ValueRef) -> ValueRef {
    if cx.unreachable { return _Undef(LHS); }
    count_insn(cx, "nuwsub");
    return llvm::LLVMBuildNUWSub(B(cx), LHS, RHS, noname());
}

fn FSub(cx: block, LHS: ValueRef, RHS: ValueRef) -> ValueRef {
    if cx.unreachable { return _Undef(LHS); }
    count_insn(cx, "sub");
    return llvm::LLVMBuildFSub(B(cx), LHS, RHS, noname());
}

fn Mul(cx: block, LHS: ValueRef, RHS: ValueRef) -> ValueRef {
    if cx.unreachable { return _Undef(LHS); }
    count_insn(cx, "mul");
    return llvm::LLVMBuildMul(B(cx), LHS, RHS, noname());
}

fn NSWMul(cx: block, LHS: ValueRef, RHS: ValueRef) -> ValueRef {
    if cx.unreachable { return _Undef(LHS); }
    count_insn(cx, "nswmul");
    return llvm::LLVMBuildNSWMul(B(cx), LHS, RHS, noname());
}

fn NUWMul(cx: block, LHS: ValueRef, RHS: ValueRef) -> ValueRef {
    if cx.unreachable { return _Undef(LHS); }
    count_insn(cx, "nuwmul");
    return llvm::LLVMBuildNUWMul(B(cx), LHS, RHS, noname());
}

fn FMul(cx: block, LHS: ValueRef, RHS: ValueRef) -> ValueRef {
    if cx.unreachable { return _Undef(LHS); }
    count_insn(cx, "fmul");
    return llvm::LLVMBuildFMul(B(cx), LHS, RHS, noname());
}

fn UDiv(cx: block, LHS: ValueRef, RHS: ValueRef) -> ValueRef {
    if cx.unreachable { return _Undef(LHS); }
    count_insn(cx, "udiv");
    return llvm::LLVMBuildUDiv(B(cx), LHS, RHS, noname());
}

fn SDiv(cx: block, LHS: ValueRef, RHS: ValueRef) -> ValueRef {
    if cx.unreachable { return _Undef(LHS); }
    count_insn(cx, "sdiv");
    return llvm::LLVMBuildSDiv(B(cx), LHS, RHS, noname());
}

fn ExactSDiv(cx: block, LHS: ValueRef, RHS: ValueRef) -> ValueRef {
    if cx.unreachable { return _Undef(LHS); }
    count_insn(cx, "extractsdiv");
    return llvm::LLVMBuildExactSDiv(B(cx), LHS, RHS, noname());
}

fn FDiv(cx: block, LHS: ValueRef, RHS: ValueRef) -> ValueRef {
    if cx.unreachable { return _Undef(LHS); }
    count_insn(cx, "fdiv");
    return llvm::LLVMBuildFDiv(B(cx), LHS, RHS, noname());
}

fn URem(cx: block, LHS: ValueRef, RHS: ValueRef) -> ValueRef {
    if cx.unreachable { return _Undef(LHS); }
    count_insn(cx, "urem");
    return llvm::LLVMBuildURem(B(cx), LHS, RHS, noname());
}

fn SRem(cx: block, LHS: ValueRef, RHS: ValueRef) -> ValueRef {
    if cx.unreachable { return _Undef(LHS); }
    count_insn(cx, "srem");
    return llvm::LLVMBuildSRem(B(cx), LHS, RHS, noname());
}

fn FRem(cx: block, LHS: ValueRef, RHS: ValueRef) -> ValueRef {
    if cx.unreachable { return _Undef(LHS); }
    count_insn(cx, "frem");
    return llvm::LLVMBuildFRem(B(cx), LHS, RHS, noname());
}

fn Shl(cx: block, LHS: ValueRef, RHS: ValueRef) -> ValueRef {
    if cx.unreachable { return _Undef(LHS); }
    count_insn(cx, "shl");
    return llvm::LLVMBuildShl(B(cx), LHS, RHS, noname());
}

fn LShr(cx: block, LHS: ValueRef, RHS: ValueRef) -> ValueRef {
    if cx.unreachable { return _Undef(LHS); }
    count_insn(cx, "lshr");
    return llvm::LLVMBuildLShr(B(cx), LHS, RHS, noname());
}

fn AShr(cx: block, LHS: ValueRef, RHS: ValueRef) -> ValueRef {
    if cx.unreachable { return _Undef(LHS); }
    count_insn(cx, "ashr");
    return llvm::LLVMBuildAShr(B(cx), LHS, RHS, noname());
}

fn And(cx: block, LHS: ValueRef, RHS: ValueRef) -> ValueRef {
    if cx.unreachable { return _Undef(LHS); }
    count_insn(cx, "and");
    return llvm::LLVMBuildAnd(B(cx), LHS, RHS, noname());
}

fn Or(cx: block, LHS: ValueRef, RHS: ValueRef) -> ValueRef {
    if cx.unreachable { return _Undef(LHS); }
    count_insn(cx, "or");
    return llvm::LLVMBuildOr(B(cx), LHS, RHS, noname());
}

fn Xor(cx: block, LHS: ValueRef, RHS: ValueRef) -> ValueRef {
    if cx.unreachable { return _Undef(LHS); }
    count_insn(cx, "xor");
    return llvm::LLVMBuildXor(B(cx), LHS, RHS, noname());
}

fn BinOp(cx: block, Op: Opcode, LHS: ValueRef, RHS: ValueRef) ->
   ValueRef {
    if cx.unreachable { return _Undef(LHS); }
    count_insn(cx, "binop");
    return llvm::LLVMBuildBinOp(B(cx), Op, LHS, RHS, noname());
}

fn Neg(cx: block, V: ValueRef) -> ValueRef {
    if cx.unreachable { return _Undef(V); }
    count_insn(cx, "neg");
    return llvm::LLVMBuildNeg(B(cx), V, noname());
}

fn NSWNeg(cx: block, V: ValueRef) -> ValueRef {
    if cx.unreachable { return _Undef(V); }
    count_insn(cx, "nswneg");
    return llvm::LLVMBuildNSWNeg(B(cx), V, noname());
}

fn NUWNeg(cx: block, V: ValueRef) -> ValueRef {
    if cx.unreachable { return _Undef(V); }
    count_insn(cx, "nuwneg");
    return llvm::LLVMBuildNUWNeg(B(cx), V, noname());
}
fn FNeg(cx: block, V: ValueRef) -> ValueRef {
    if cx.unreachable { return _Undef(V); }
    count_insn(cx, "fneg");
    return llvm::LLVMBuildFNeg(B(cx), V, noname());
}

fn Not(cx: block, V: ValueRef) -> ValueRef {
    if cx.unreachable { return _Undef(V); }
    count_insn(cx, "not");
    return llvm::LLVMBuildNot(B(cx), V, noname());
}

/* Memory */
fn Malloc(cx: block, Ty: TypeRef) -> ValueRef {
    if cx.unreachable { return llvm::LLVMGetUndef(T_ptr(T_i8())); }
    count_insn(cx, "malloc");
    return llvm::LLVMBuildMalloc(B(cx), Ty, noname());
}

fn ArrayMalloc(cx: block, Ty: TypeRef, Val: ValueRef) -> ValueRef {
    if cx.unreachable { return llvm::LLVMGetUndef(T_ptr(T_i8())); }
    count_insn(cx, "arraymalloc");
    return llvm::LLVMBuildArrayMalloc(B(cx), Ty, Val, noname());
}

fn Alloca(cx: block, Ty: TypeRef) -> ValueRef {
    if cx.unreachable { return llvm::LLVMGetUndef(T_ptr(Ty)); }
    count_insn(cx, "alloca");
    return llvm::LLVMBuildAlloca(B(cx), Ty, noname());
}

fn ArrayAlloca(cx: block, Ty: TypeRef, Val: ValueRef) -> ValueRef {
    if cx.unreachable { return llvm::LLVMGetUndef(T_ptr(Ty)); }
    count_insn(cx, "arrayalloca");
    return llvm::LLVMBuildArrayAlloca(B(cx), Ty, Val, noname());
}

fn Free(cx: block, PointerVal: ValueRef) {
    if cx.unreachable { return; }
    count_insn(cx, "free");
    llvm::LLVMBuildFree(B(cx), PointerVal);
}

fn Load(cx: block, PointerVal: ValueRef) -> ValueRef {
    let ccx = cx.fcx.ccx;
    if cx.unreachable {
        let ty = val_ty(PointerVal);
        let eltty = if llvm::LLVMGetTypeKind(ty) == lib::llvm::Array {
            llvm::LLVMGetElementType(ty) } else { ccx.int_type };
        return llvm::LLVMGetUndef(eltty);
    }
    count_insn(cx, "load");
    return llvm::LLVMBuildLoad(B(cx), PointerVal, noname());
}

fn Store(cx: block, Val: ValueRef, Ptr: ValueRef) {
    if cx.unreachable { return; }
    debug!("Store %s -> %s",
           val_str(cx.ccx().tn, Val),
           val_str(cx.ccx().tn, Ptr));
    count_insn(cx, "store");
    llvm::LLVMBuildStore(B(cx), Val, Ptr);
}

fn GEP(cx: block, Pointer: ValueRef, Indices: ~[ValueRef]) -> ValueRef {
    if cx.unreachable { return llvm::LLVMGetUndef(T_ptr(T_nil())); }
    unsafe {
    count_insn(cx, "gep");
    return llvm::LLVMBuildGEP(B(cx), Pointer, vec::unsafe::to_ptr(Indices),
                               Indices.len() as c_uint, noname());
    }
}

// Simple wrapper around GEP that takes an array of ints and wraps them
// in C_i32()
//
// XXX: Use a small-vector optimization to avoid allocations here.
fn GEPi(cx: block, base: ValueRef, ixs: &[uint]) -> ValueRef {
    let mut v: ~[ValueRef] = ~[];
    for vec::each(ixs) |i| { vec::push(v, C_i32(i as i32)); }
    count_insn(cx, "gepi");
    return InBoundsGEP(cx, base, v);
}

fn InBoundsGEP(cx: block, Pointer: ValueRef, Indices: &[ValueRef]) ->
   ValueRef {
    if cx.unreachable { return llvm::LLVMGetUndef(T_ptr(T_nil())); }
    unsafe {
        count_insn(cx, "inboundsgep");
    return llvm::LLVMBuildInBoundsGEP(B(cx), Pointer,
                                       vec::unsafe::to_ptr(Indices),
                                       Indices.len() as c_uint,
                                       noname());
    }
}

fn StructGEP(cx: block, Pointer: ValueRef, Idx: uint) -> ValueRef {
    if cx.unreachable { return llvm::LLVMGetUndef(T_ptr(T_nil())); }
    count_insn(cx, "structgep");
    return llvm::LLVMBuildStructGEP(B(cx), Pointer, Idx as c_uint, noname());
}

fn GlobalString(cx: block, _Str: *libc::c_char) -> ValueRef {
    if cx.unreachable { return llvm::LLVMGetUndef(T_ptr(T_i8())); }
    count_insn(cx, "globalstring");
    return llvm::LLVMBuildGlobalString(B(cx), _Str, noname());
}

fn GlobalStringPtr(cx: block, _Str: *libc::c_char) -> ValueRef {
    if cx.unreachable { return llvm::LLVMGetUndef(T_ptr(T_i8())); }
    count_insn(cx, "globalstringptr");
    return llvm::LLVMBuildGlobalStringPtr(B(cx), _Str, noname());
}

/* Casts */
fn Trunc(cx: block, Val: ValueRef, DestTy: TypeRef) -> ValueRef {
    if cx.unreachable { return llvm::LLVMGetUndef(DestTy); }
    count_insn(cx, "trunc");
    return llvm::LLVMBuildTrunc(B(cx), Val, DestTy, noname());
}

fn ZExt(cx: block, Val: ValueRef, DestTy: TypeRef) -> ValueRef {
    if cx.unreachable { return llvm::LLVMGetUndef(DestTy); }
    count_insn(cx, "zext");
    return llvm::LLVMBuildZExt(B(cx), Val, DestTy, noname());
}

fn SExt(cx: block, Val: ValueRef, DestTy: TypeRef) -> ValueRef {
    if cx.unreachable { return llvm::LLVMGetUndef(DestTy); }
    count_insn(cx, "sext");
    return llvm::LLVMBuildSExt(B(cx), Val, DestTy, noname());
}

fn FPToUI(cx: block, Val: ValueRef, DestTy: TypeRef) -> ValueRef {
    if cx.unreachable { return llvm::LLVMGetUndef(DestTy); }
    count_insn(cx, "fptoui");
    return llvm::LLVMBuildFPToUI(B(cx), Val, DestTy, noname());
}

fn FPToSI(cx: block, Val: ValueRef, DestTy: TypeRef) -> ValueRef {
    if cx.unreachable { return llvm::LLVMGetUndef(DestTy); }
    count_insn(cx, "fptosi");
    return llvm::LLVMBuildFPToSI(B(cx), Val, DestTy, noname());
}

fn UIToFP(cx: block, Val: ValueRef, DestTy: TypeRef) -> ValueRef {
    if cx.unreachable { return llvm::LLVMGetUndef(DestTy); }
    count_insn(cx, "uitofp");
    return llvm::LLVMBuildUIToFP(B(cx), Val, DestTy, noname());
}

fn SIToFP(cx: block, Val: ValueRef, DestTy: TypeRef) -> ValueRef {
    if cx.unreachable { return llvm::LLVMGetUndef(DestTy); }
    count_insn(cx, "sitofp");
    return llvm::LLVMBuildSIToFP(B(cx), Val, DestTy, noname());
}

fn FPTrunc(cx: block, Val: ValueRef, DestTy: TypeRef) -> ValueRef {
    if cx.unreachable { return llvm::LLVMGetUndef(DestTy); }
    count_insn(cx, "fptrunc");
    return llvm::LLVMBuildFPTrunc(B(cx), Val, DestTy, noname());
}

fn FPExt(cx: block, Val: ValueRef, DestTy: TypeRef) -> ValueRef {
    if cx.unreachable { return llvm::LLVMGetUndef(DestTy); }
    count_insn(cx, "fpext");
    return llvm::LLVMBuildFPExt(B(cx), Val, DestTy, noname());
}

fn PtrToInt(cx: block, Val: ValueRef, DestTy: TypeRef) -> ValueRef {
    if cx.unreachable { return llvm::LLVMGetUndef(DestTy); }
    count_insn(cx, "ptrtoint");
    return llvm::LLVMBuildPtrToInt(B(cx), Val, DestTy, noname());
}

fn IntToPtr(cx: block, Val: ValueRef, DestTy: TypeRef) -> ValueRef {
    if cx.unreachable { return llvm::LLVMGetUndef(DestTy); }
    count_insn(cx, "inttoptr");
    return llvm::LLVMBuildIntToPtr(B(cx), Val, DestTy, noname());
}

fn BitCast(cx: block, Val: ValueRef, DestTy: TypeRef) -> ValueRef {
    if cx.unreachable { return llvm::LLVMGetUndef(DestTy); }
    count_insn(cx, "bitcast");
    return llvm::LLVMBuildBitCast(B(cx), Val, DestTy, noname());
}

fn ZExtOrBitCast(cx: block, Val: ValueRef, DestTy: TypeRef) ->
   ValueRef {
    if cx.unreachable { return llvm::LLVMGetUndef(DestTy); }
    count_insn(cx, "zextorbitcast");
    return llvm::LLVMBuildZExtOrBitCast(B(cx), Val, DestTy, noname());
}

fn SExtOrBitCast(cx: block, Val: ValueRef, DestTy: TypeRef) ->
   ValueRef {
    if cx.unreachable { return llvm::LLVMGetUndef(DestTy); }
    count_insn(cx, "sextorbitcast");
    return llvm::LLVMBuildSExtOrBitCast(B(cx), Val, DestTy, noname());
}

fn TruncOrBitCast(cx: block, Val: ValueRef, DestTy: TypeRef) ->
   ValueRef {
    if cx.unreachable { return llvm::LLVMGetUndef(DestTy); }
    count_insn(cx, "truncorbitcast");
    return llvm::LLVMBuildTruncOrBitCast(B(cx), Val, DestTy, noname());
}

fn Cast(cx: block, Op: Opcode, Val: ValueRef, DestTy: TypeRef,
        _Name: *u8) -> ValueRef {
    if cx.unreachable { return llvm::LLVMGetUndef(DestTy); }
    count_insn(cx, "cast");
    return llvm::LLVMBuildCast(B(cx), Op, Val, DestTy, noname());
}

fn PointerCast(cx: block, Val: ValueRef, DestTy: TypeRef) -> ValueRef {
    if cx.unreachable { return llvm::LLVMGetUndef(DestTy); }
    count_insn(cx, "pointercast");
    return llvm::LLVMBuildPointerCast(B(cx), Val, DestTy, noname());
}

fn IntCast(cx: block, Val: ValueRef, DestTy: TypeRef) -> ValueRef {
    if cx.unreachable { return llvm::LLVMGetUndef(DestTy); }
    count_insn(cx, "intcast");
    return llvm::LLVMBuildIntCast(B(cx), Val, DestTy, noname());
}

fn FPCast(cx: block, Val: ValueRef, DestTy: TypeRef) -> ValueRef {
    if cx.unreachable { return llvm::LLVMGetUndef(DestTy); }
    count_insn(cx, "fpcast");
    return llvm::LLVMBuildFPCast(B(cx), Val, DestTy, noname());
}


/* Comparisons */
fn ICmp(cx: block, Op: IntPredicate, LHS: ValueRef, RHS: ValueRef)
    -> ValueRef {
    if cx.unreachable { return llvm::LLVMGetUndef(T_i1()); }
    count_insn(cx, "icmp");
    return llvm::LLVMBuildICmp(B(cx), Op as c_uint, LHS, RHS, noname());
}

fn FCmp(cx: block, Op: RealPredicate, LHS: ValueRef, RHS: ValueRef)
    -> ValueRef {
    if cx.unreachable { return llvm::LLVMGetUndef(T_i1()); }
    count_insn(cx, "fcmp");
    return llvm::LLVMBuildFCmp(B(cx), Op as c_uint, LHS, RHS, noname());
}

/* Miscellaneous instructions */
fn EmptyPhi(cx: block, Ty: TypeRef) -> ValueRef {
    if cx.unreachable { return llvm::LLVMGetUndef(Ty); }
    count_insn(cx, "emptyphi");
    return llvm::LLVMBuildPhi(B(cx), Ty, noname());
}

fn Phi(cx: block, Ty: TypeRef, vals: ~[ValueRef], bbs: ~[BasicBlockRef])
   -> ValueRef {
    if cx.unreachable { return llvm::LLVMGetUndef(Ty); }
    assert vals.len() == bbs.len();
    let phi = EmptyPhi(cx, Ty);
    unsafe {
        count_insn(cx, "addincoming");
        llvm::LLVMAddIncoming(phi, vec::unsafe::to_ptr(vals),
                              vec::unsafe::to_ptr(bbs),
                              vals.len() as c_uint);
        return phi;
    }
}

fn AddIncomingToPhi(phi: ValueRef, val: ValueRef, bb: BasicBlockRef) {
    if llvm::LLVMIsUndef(phi) == lib::llvm::True { return; }
    unsafe {
        let valptr = unsafe::reinterpret_cast(&ptr::addr_of(val));
        let bbptr = unsafe::reinterpret_cast(&ptr::addr_of(bb));
        llvm::LLVMAddIncoming(phi, valptr, bbptr, 1 as c_uint);
    }
}

fn _UndefReturn(cx: block, Fn: ValueRef) -> ValueRef {
    let ccx = cx.fcx.ccx;
    let ty = val_ty(Fn);
    let retty = if llvm::LLVMGetTypeKind(ty) == lib::llvm::Integer {
        llvm::LLVMGetReturnType(ty) } else { ccx.int_type };
        count_insn(cx, ~"");
    return llvm::LLVMGetUndef(retty);
}

fn add_span_comment(bcx: block, sp: span, text: ~str) {
    let ccx = bcx.ccx();
    if !ccx.sess.no_asm_comments() {
        let s = text + ~" (" + codemap::span_to_str(sp, ccx.sess.codemap)
            + ~")";
        log(debug, s);
        add_comment(bcx, s);
    }
}

fn add_comment(bcx: block, text: ~str) {
    let ccx = bcx.ccx();
    if !ccx.sess.no_asm_comments() {
        let sanitized = str::replace(text, ~"$", ~"");
        let comment_text = ~"# " + str::replace(sanitized, ~"\n", ~"\n\t# ");
        let asm = str::as_c_str(comment_text, |c| {
            str::as_c_str(~"", |e| {
                count_insn(bcx, ~"inlineasm");
                llvm::LLVMConstInlineAsm(T_fn(~[], T_void()), c, e,
                                         False, False)
            })
        });
        Call(bcx, asm, ~[]);
    }
}

fn Call(cx: block, Fn: ValueRef, Args: &[ValueRef]) -> ValueRef {
    if cx.unreachable { return _UndefReturn(cx, Fn); }
    unsafe {
        count_insn(cx, "call");

        debug!("Call(Fn=%s, Args=%?)",
               val_str(cx.ccx().tn, Fn),
               Args.map(|arg| val_str(cx.ccx().tn, arg)));

        do vec::as_buf(Args) |ptr, len| {
            llvm::LLVMBuildCall(B(cx), Fn, ptr, len as c_uint, noname())
        }
    }
}

fn FastCall(cx: block, Fn: ValueRef, Args: ~[ValueRef]) -> ValueRef {
    if cx.unreachable { return _UndefReturn(cx, Fn); }
    unsafe {
        count_insn(cx, "fastcall");
        let v = llvm::LLVMBuildCall(B(cx), Fn, vec::unsafe::to_ptr(Args),
                                    Args.len() as c_uint, noname());
        lib::llvm::SetInstructionCallConv(v, lib::llvm::FastCallConv);
        return v;
    }
}

fn CallWithConv(cx: block, Fn: ValueRef, Args: ~[ValueRef],
                Conv: CallConv) -> ValueRef {
    if cx.unreachable { return _UndefReturn(cx, Fn); }
    unsafe {
        count_insn(cx, "callwithconv");
        let v = llvm::LLVMBuildCall(B(cx), Fn, vec::unsafe::to_ptr(Args),
                                    Args.len() as c_uint, noname());
        lib::llvm::SetInstructionCallConv(v, Conv);
        return v;
    }
}

fn Select(cx: block, If: ValueRef, Then: ValueRef, Else: ValueRef) ->
   ValueRef {
    if cx.unreachable { return _Undef(Then); }
    count_insn(cx, "select");
    return llvm::LLVMBuildSelect(B(cx), If, Then, Else, noname());
}

fn VAArg(cx: block, list: ValueRef, Ty: TypeRef) -> ValueRef {
    if cx.unreachable { return llvm::LLVMGetUndef(Ty); }
    count_insn(cx, "vaarg");
    return llvm::LLVMBuildVAArg(B(cx), list, Ty, noname());
}

fn ExtractElement(cx: block, VecVal: ValueRef, Index: ValueRef) ->
   ValueRef {
    if cx.unreachable { return llvm::LLVMGetUndef(T_nil()); }
    count_insn(cx, "extractelement");
    return llvm::LLVMBuildExtractElement(B(cx), VecVal, Index, noname());
}

fn InsertElement(cx: block, VecVal: ValueRef, EltVal: ValueRef,
                 Index: ValueRef) {
    if cx.unreachable { return; }
    count_insn(cx, "insertelement");
    llvm::LLVMBuildInsertElement(B(cx), VecVal, EltVal, Index, noname());
}

fn ShuffleVector(cx: block, V1: ValueRef, V2: ValueRef,
                 Mask: ValueRef) {
    if cx.unreachable { return; }
    count_insn(cx, "shufflevector");
    llvm::LLVMBuildShuffleVector(B(cx), V1, V2, Mask, noname());
}

fn ExtractValue(cx: block, AggVal: ValueRef, Index: uint) -> ValueRef {
    if cx.unreachable { return llvm::LLVMGetUndef(T_nil()); }
    count_insn(cx, "extractvalue");
    return llvm::LLVMBuildExtractValue(
        B(cx), AggVal, Index as c_uint, noname());
}

fn InsertValue(cx: block, AggVal: ValueRef, EltVal: ValueRef,
               Index: uint) {
    if cx.unreachable { return; }
    count_insn(cx, "insertvalue");
    llvm::LLVMBuildInsertValue(B(cx), AggVal, EltVal, Index as c_uint,
                               noname());
}

fn IsNull(cx: block, Val: ValueRef) -> ValueRef {
    if cx.unreachable { return llvm::LLVMGetUndef(T_i1()); }
    count_insn(cx, "isnull");
    return llvm::LLVMBuildIsNull(B(cx), Val, noname());
}

fn IsNotNull(cx: block, Val: ValueRef) -> ValueRef {
    if cx.unreachable { return llvm::LLVMGetUndef(T_i1()); }
    count_insn(cx, "isnotnull");
    return llvm::LLVMBuildIsNotNull(B(cx), Val, noname());
}

fn PtrDiff(cx: block, LHS: ValueRef, RHS: ValueRef) -> ValueRef {
    let ccx = cx.fcx.ccx;
    if cx.unreachable { return llvm::LLVMGetUndef(ccx.int_type); }
    count_insn(cx, "ptrdiff");
    return llvm::LLVMBuildPtrDiff(B(cx), LHS, RHS, noname());
}

fn Trap(cx: block) {
    if cx.unreachable { return; }
    let b = B(cx);
    let BB: BasicBlockRef = llvm::LLVMGetInsertBlock(b);
    let FN: ValueRef = llvm::LLVMGetBasicBlockParent(BB);
    let M: ModuleRef = llvm::LLVMGetGlobalParent(FN);
    let T: ValueRef = str::as_c_str(~"llvm.trap", |buf| {
        llvm::LLVMGetNamedFunction(M, buf)
    });
    assert (T as int != 0);
    let Args: ~[ValueRef] = ~[];
    unsafe {
        count_insn(cx, "trap");
        llvm::LLVMBuildCall(b, T, vec::unsafe::to_ptr(Args),
                            Args.len() as c_uint, noname());
    }
}

fn LandingPad(cx: block, Ty: TypeRef, PersFn: ValueRef,
              NumClauses: uint) -> ValueRef {
    assert !cx.terminated && !cx.unreachable;
    count_insn(cx, "landingpad");
    return llvm::LLVMBuildLandingPad(B(cx), Ty, PersFn,
                                  NumClauses as c_uint, noname());
}

fn SetCleanup(cx: block, LandingPad: ValueRef) {
    count_insn(cx, "setcleanup");
    llvm::LLVMSetCleanup(LandingPad, lib::llvm::True);
}

fn Resume(cx: block, Exn: ValueRef) -> ValueRef {
    assert (!cx.terminated);
    cx.terminated = true;
    count_insn(cx, "resume");
    return llvm::LLVMBuildResume(B(cx), Exn);
}

// Atomic Operations
fn AtomicRMW(cx: block, op: AtomicBinOp,
             dst: ValueRef, src: ValueRef,
             order: AtomicOrdering) -> ValueRef {
    llvm::LLVMBuildAtomicRMW(B(cx), op, dst, src, order)
}

//
// Local Variables:
// mode: rust
// fill-column: 78;
// indent-tabs-mode: nil
// c-basic-offset: 4
// buffer-file-coding-system: utf-8-unix
// End:
//
