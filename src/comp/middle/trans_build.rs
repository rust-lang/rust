import core::{vec, str};
import str::sbuf;
import lib::llvm::llvm;
import syntax::codemap::span;
import llvm::{ValueRef, TypeRef, BasicBlockRef, BuilderRef, Opcode,
              ModuleRef};
import trans_common::{block_ctxt, T_ptr, T_nil, T_i8, T_i1, T_void,
                      T_fn, val_ty, bcx_ccx, C_i32};

fn B(cx: @block_ctxt) -> BuilderRef {
    let b = *cx.fcx.lcx.ccx.builder;
    llvm::LLVMPositionBuilderAtEnd(b, cx.llbb);
    ret b;
}

// The difference between a block being unreachable and being terminated is
// somewhat obscure, and has to do with error checking. When a block is
// terminated, we're saying that trying to add any further statements in the
// block is an error. On the other hand, if something is unreachable, that
// means that the block was terminated in some way that we don't want to check
// for (fail/break/ret statements, call to diverging functions, etc), and
// further instructions to the block should simply be ignored.

fn RetVoid(cx: @block_ctxt) {
    if cx.unreachable { ret; }
    assert (!cx.terminated);
    cx.terminated = true;
    let instr = llvm::LLVMBuildRetVoid(B(cx));
    debuginfo::add_line_info(cx, instr);
}

fn Ret(cx: @block_ctxt, V: ValueRef) {
    if cx.unreachable { ret; }
    assert (!cx.terminated);
    cx.terminated = true;
    let instr = llvm::LLVMBuildRet(B(cx), V);
    debuginfo::add_line_info(cx, instr);
}

fn AggregateRet(cx: @block_ctxt, RetVals: [ValueRef]) {
    if cx.unreachable { ret; }
    assert (!cx.terminated);
    cx.terminated = true;
    unsafe {
        let instr = llvm::LLVMBuildAggregateRet(B(cx), vec::to_ptr(RetVals),
                                                vec::len(RetVals));
        debuginfo::add_line_info(cx, instr);
    }
}

fn Br(cx: @block_ctxt, Dest: BasicBlockRef) {
    if cx.unreachable { ret; }
    assert (!cx.terminated);
    cx.terminated = true;
    let instr = llvm::LLVMBuildBr(B(cx), Dest);
    debuginfo::add_line_info(cx, instr);
}

fn CondBr(cx: @block_ctxt, If: ValueRef, Then: BasicBlockRef,
          Else: BasicBlockRef) {
    if cx.unreachable { ret; }
    assert (!cx.terminated);
    cx.terminated = true;
    let instr = llvm::LLVMBuildCondBr(B(cx), If, Then, Else);
    debuginfo::add_line_info(cx, instr);
}

fn Switch(cx: @block_ctxt, V: ValueRef, Else: BasicBlockRef, NumCases: uint)
    -> ValueRef {
    if cx.unreachable { ret _Undef(V); }
    assert !cx.terminated;
    cx.terminated = true;
    let instr = llvm::LLVMBuildSwitch(B(cx), V, Else, NumCases);
    debuginfo::add_line_info(cx, instr);
    ret instr;
}

fn AddCase(S: ValueRef, OnVal: ValueRef, Dest: BasicBlockRef) {
    if llvm::LLVMIsUndef(S) == lib::llvm::True { ret; }
    llvm::LLVMAddCase(S, OnVal, Dest);
}

fn IndirectBr(cx: @block_ctxt, Addr: ValueRef, NumDests: uint) {
    if cx.unreachable { ret; }
    assert (!cx.terminated);
    cx.terminated = true;
    let instr = llvm::LLVMBuildIndirectBr(B(cx), Addr, NumDests);
    debuginfo::add_line_info(cx, instr);
}

// This is a really awful way to get a zero-length c-string, but better (and a
// lot more efficient) than doing str::as_buf("", ...) every time.
fn noname() -> sbuf unsafe {
    const cnull: uint = 0u;
    ret unsafe::reinterpret_cast(ptr::addr_of(cnull));
}

fn Invoke(cx: @block_ctxt, Fn: ValueRef, Args: [ValueRef],
          Then: BasicBlockRef, Catch: BasicBlockRef) {
    if cx.unreachable { ret; }
    assert (!cx.terminated);
    cx.terminated = true;
    unsafe {
        let instr = llvm::LLVMBuildInvoke(B(cx), Fn, vec::to_ptr(Args),
                                          vec::len(Args), Then, Catch,
                                          noname());
        debuginfo::add_line_info(cx, instr);
    }
}

fn FastInvoke(cx: @block_ctxt, Fn: ValueRef, Args: [ValueRef],
              Then: BasicBlockRef, Catch: BasicBlockRef) {
    if cx.unreachable { ret; }
    assert (!cx.terminated);
    cx.terminated = true;
    unsafe {
        let v = llvm::LLVMBuildInvoke(B(cx), Fn, vec::to_ptr(Args),
                                      vec::len(Args), Then, Catch, noname());
        llvm::LLVMSetInstructionCallConv(v, lib::llvm::LLVMFastCallConv);
        debuginfo::add_line_info(cx, v);
    }
}

fn Unreachable(cx: @block_ctxt) {
    if cx.unreachable { ret; }
    cx.unreachable = true;
    if !cx.terminated { llvm::LLVMBuildUnreachable(B(cx)); }
}

fn _Undef(val: ValueRef) -> ValueRef {
    ret llvm::LLVMGetUndef(val_ty(val));
}

/* Arithmetic */
fn Add(cx: @block_ctxt, LHS: ValueRef, RHS: ValueRef) -> ValueRef {
    if cx.unreachable { ret _Undef(LHS); }
    let instr = llvm::LLVMBuildAdd(B(cx), LHS, RHS, noname());
    debuginfo::add_line_info(cx, instr);
    ret instr;
}

fn NSWAdd(cx: @block_ctxt, LHS: ValueRef, RHS: ValueRef) -> ValueRef {
    if cx.unreachable { ret _Undef(LHS); }
    let instr = llvm::LLVMBuildNSWAdd(B(cx), LHS, RHS, noname());
    debuginfo::add_line_info(cx, instr);
    ret instr;
}

fn NUWAdd(cx: @block_ctxt, LHS: ValueRef, RHS: ValueRef) -> ValueRef {
    if cx.unreachable { ret _Undef(LHS); }
    let instr = llvm::LLVMBuildNUWAdd(B(cx), LHS, RHS, noname());
    debuginfo::add_line_info(cx, instr);
    ret instr;
}

fn FAdd(cx: @block_ctxt, LHS: ValueRef, RHS: ValueRef) -> ValueRef {
    if cx.unreachable { ret _Undef(LHS); }
    let instr = llvm::LLVMBuildFAdd(B(cx), LHS, RHS, noname());
    debuginfo::add_line_info(cx, instr);
    ret instr;
}

fn Sub(cx: @block_ctxt, LHS: ValueRef, RHS: ValueRef) -> ValueRef {
    if cx.unreachable { ret _Undef(LHS); }
    let instr = llvm::LLVMBuildSub(B(cx), LHS, RHS, noname());
    debuginfo::add_line_info(cx, instr);
    ret instr;
}

fn NSWSub(cx: @block_ctxt, LHS: ValueRef, RHS: ValueRef) -> ValueRef {
    if cx.unreachable { ret _Undef(LHS); }
    let instr = llvm::LLVMBuildNSWSub(B(cx), LHS, RHS, noname());
    debuginfo::add_line_info(cx, instr);
    ret instr;
}

fn NUWSub(cx: @block_ctxt, LHS: ValueRef, RHS: ValueRef) -> ValueRef {
    if cx.unreachable { ret _Undef(LHS); }
    let instr = llvm::LLVMBuildNUWSub(B(cx), LHS, RHS, noname());
    debuginfo::add_line_info(cx, instr);
    ret instr;
}

fn FSub(cx: @block_ctxt, LHS: ValueRef, RHS: ValueRef) -> ValueRef {
    if cx.unreachable { ret _Undef(LHS); }
    let instr = llvm::LLVMBuildFSub(B(cx), LHS, RHS, noname());
    debuginfo::add_line_info(cx, instr);
    ret instr;
}

fn Mul(cx: @block_ctxt, LHS: ValueRef, RHS: ValueRef) -> ValueRef {
    if cx.unreachable { ret _Undef(LHS); }
    let instr = llvm::LLVMBuildMul(B(cx), LHS, RHS, noname());
    debuginfo::add_line_info(cx, instr);
    ret instr;
}

fn NSWMul(cx: @block_ctxt, LHS: ValueRef, RHS: ValueRef) -> ValueRef {
    if cx.unreachable { ret _Undef(LHS); }
    let instr = llvm::LLVMBuildNSWMul(B(cx), LHS, RHS, noname());
    debuginfo::add_line_info(cx, instr);
    ret instr;
}

fn NUWMul(cx: @block_ctxt, LHS: ValueRef, RHS: ValueRef) -> ValueRef {
    if cx.unreachable { ret _Undef(LHS); }
    let instr = llvm::LLVMBuildNUWMul(B(cx), LHS, RHS, noname());
    debuginfo::add_line_info(cx, instr);
    ret instr;
}

fn FMul(cx: @block_ctxt, LHS: ValueRef, RHS: ValueRef) -> ValueRef {
    if cx.unreachable { ret _Undef(LHS); }
    let instr = llvm::LLVMBuildFMul(B(cx), LHS, RHS, noname());
    debuginfo::add_line_info(cx, instr);
    ret instr;
}

fn UDiv(cx: @block_ctxt, LHS: ValueRef, RHS: ValueRef) -> ValueRef {
    if cx.unreachable { ret _Undef(LHS); }
    let instr = llvm::LLVMBuildUDiv(B(cx), LHS, RHS, noname());
    debuginfo::add_line_info(cx, instr);
    ret instr;
}

fn SDiv(cx: @block_ctxt, LHS: ValueRef, RHS: ValueRef) -> ValueRef {
    if cx.unreachable { ret _Undef(LHS); }
    let instr = llvm::LLVMBuildSDiv(B(cx), LHS, RHS, noname());
    debuginfo::add_line_info(cx, instr);
    ret instr;
}

fn ExactSDiv(cx: @block_ctxt, LHS: ValueRef, RHS: ValueRef) -> ValueRef {
    if cx.unreachable { ret _Undef(LHS); }
    let instr = llvm::LLVMBuildExactSDiv(B(cx), LHS, RHS, noname());
    debuginfo::add_line_info(cx, instr);
    ret instr;
}

fn FDiv(cx: @block_ctxt, LHS: ValueRef, RHS: ValueRef) -> ValueRef {
    if cx.unreachable { ret _Undef(LHS); }
    let instr = llvm::LLVMBuildFDiv(B(cx), LHS, RHS, noname());
    debuginfo::add_line_info(cx, instr);
    ret instr;
}

fn URem(cx: @block_ctxt, LHS: ValueRef, RHS: ValueRef) -> ValueRef {
    if cx.unreachable { ret _Undef(LHS); }
    let instr = llvm::LLVMBuildURem(B(cx), LHS, RHS, noname());
    debuginfo::add_line_info(cx, instr);
    ret instr;
}

fn SRem(cx: @block_ctxt, LHS: ValueRef, RHS: ValueRef) -> ValueRef {
    if cx.unreachable { ret _Undef(LHS); }
    let instr = llvm::LLVMBuildSRem(B(cx), LHS, RHS, noname());
    debuginfo::add_line_info(cx, instr);
    ret instr;
}

fn FRem(cx: @block_ctxt, LHS: ValueRef, RHS: ValueRef) -> ValueRef {
    if cx.unreachable { ret _Undef(LHS); }
    let instr = llvm::LLVMBuildFRem(B(cx), LHS, RHS, noname());
    debuginfo::add_line_info(cx, instr);
    ret instr;
}

fn Shl(cx: @block_ctxt, LHS: ValueRef, RHS: ValueRef) -> ValueRef {
    if cx.unreachable { ret _Undef(LHS); }
    let instr = llvm::LLVMBuildShl(B(cx), LHS, RHS, noname());
    debuginfo::add_line_info(cx, instr);
    ret instr;
}

fn LShr(cx: @block_ctxt, LHS: ValueRef, RHS: ValueRef) -> ValueRef {
    if cx.unreachable { ret _Undef(LHS); }
    let instr = llvm::LLVMBuildLShr(B(cx), LHS, RHS, noname());
    debuginfo::add_line_info(cx, instr);
    ret instr;
}

fn AShr(cx: @block_ctxt, LHS: ValueRef, RHS: ValueRef) -> ValueRef {
    if cx.unreachable { ret _Undef(LHS); }
    let instr = llvm::LLVMBuildAShr(B(cx), LHS, RHS, noname());
    debuginfo::add_line_info(cx, instr);
    ret instr;
}

fn And(cx: @block_ctxt, LHS: ValueRef, RHS: ValueRef) -> ValueRef {
    if cx.unreachable { ret _Undef(LHS); }
    let instr = llvm::LLVMBuildAnd(B(cx), LHS, RHS, noname());
    debuginfo::add_line_info(cx, instr);
    ret instr;
}

fn Or(cx: @block_ctxt, LHS: ValueRef, RHS: ValueRef) -> ValueRef {
    if cx.unreachable { ret _Undef(LHS); }
    let instr = llvm::LLVMBuildOr(B(cx), LHS, RHS, noname());
    debuginfo::add_line_info(cx, instr);
    ret instr;
}

fn Xor(cx: @block_ctxt, LHS: ValueRef, RHS: ValueRef) -> ValueRef {
    if cx.unreachable { ret _Undef(LHS); }
    let instr = llvm::LLVMBuildXor(B(cx), LHS, RHS, noname());
    debuginfo::add_line_info(cx, instr);
    ret instr;
}

fn BinOp(cx: @block_ctxt, Op: Opcode, LHS: ValueRef, RHS: ValueRef) ->
   ValueRef {
    if cx.unreachable { ret _Undef(LHS); }
    let instr = llvm::LLVMBuildBinOp(B(cx), Op, LHS, RHS, noname());
    debuginfo::add_line_info(cx, instr);
    ret instr;
}

fn Neg(cx: @block_ctxt, V: ValueRef) -> ValueRef {
    if cx.unreachable { ret _Undef(V); }
    let instr = llvm::LLVMBuildNeg(B(cx), V, noname());
    debuginfo::add_line_info(cx, instr);
    ret instr;
}

fn NSWNeg(cx: @block_ctxt, V: ValueRef) -> ValueRef {
    if cx.unreachable { ret _Undef(V); }
    let instr = llvm::LLVMBuildNSWNeg(B(cx), V, noname());
    debuginfo::add_line_info(cx, instr);
    ret instr;
}

fn NUWNeg(cx: @block_ctxt, V: ValueRef) -> ValueRef {
    if cx.unreachable { ret _Undef(V); }
    let instr = llvm::LLVMBuildNUWNeg(B(cx), V, noname());
    debuginfo::add_line_info(cx, instr);
    ret instr;
}
fn FNeg(cx: @block_ctxt, V: ValueRef) -> ValueRef {
    if cx.unreachable { ret _Undef(V); }
    let instr = llvm::LLVMBuildFNeg(B(cx), V, noname());
    debuginfo::add_line_info(cx, instr);
    ret instr;
}

fn Not(cx: @block_ctxt, V: ValueRef) -> ValueRef {
    if cx.unreachable { ret _Undef(V); }
    let instr = llvm::LLVMBuildNot(B(cx), V, noname());
    debuginfo::add_line_info(cx, instr);
    ret instr;
}

/* Memory */
fn Malloc(cx: @block_ctxt, Ty: TypeRef) -> ValueRef {
    if cx.unreachable { ret llvm::LLVMGetUndef(T_ptr(T_i8())); }
    let instr = llvm::LLVMBuildMalloc(B(cx), Ty, noname());
    debuginfo::add_line_info(cx, instr);
    ret instr;
}

fn ArrayMalloc(cx: @block_ctxt, Ty: TypeRef, Val: ValueRef) -> ValueRef {
    if cx.unreachable { ret llvm::LLVMGetUndef(T_ptr(T_i8())); }
    let instr = llvm::LLVMBuildArrayMalloc(B(cx), Ty, Val, noname());
    debuginfo::add_line_info(cx, instr);
    ret instr;
}

fn Alloca(cx: @block_ctxt, Ty: TypeRef) -> ValueRef {
    if cx.unreachable { ret llvm::LLVMGetUndef(T_ptr(Ty)); }
    let instr = llvm::LLVMBuildAlloca(B(cx), Ty, noname());
    debuginfo::add_line_info(cx, instr);
    ret instr;
}

fn ArrayAlloca(cx: @block_ctxt, Ty: TypeRef, Val: ValueRef) -> ValueRef {
    if cx.unreachable { ret llvm::LLVMGetUndef(T_ptr(Ty)); }
    let instr = llvm::LLVMBuildArrayAlloca(B(cx), Ty, Val, noname());
    debuginfo::add_line_info(cx, instr);
    ret instr;
}

fn Free(cx: @block_ctxt, PointerVal: ValueRef) {
    if cx.unreachable { ret; }
    let instr = llvm::LLVMBuildFree(B(cx), PointerVal);
    debuginfo::add_line_info(cx, instr);
}

fn Load(cx: @block_ctxt, PointerVal: ValueRef) -> ValueRef {
    let ccx = cx.fcx.lcx.ccx;
    if cx.unreachable {
        let ty = val_ty(PointerVal);
        let eltty = if llvm::LLVMGetTypeKind(ty) == 11 {
            llvm::LLVMGetElementType(ty) } else { ccx.int_type };
        ret llvm::LLVMGetUndef(eltty);
    }
    let instr = llvm::LLVMBuildLoad(B(cx), PointerVal, noname());
    debuginfo::add_line_info(cx, instr);
    ret instr;
}

fn Store(cx: @block_ctxt, Val: ValueRef, Ptr: ValueRef) {
    if cx.unreachable { ret; }
    let instr = llvm::LLVMBuildStore(B(cx), Val, Ptr);
    debuginfo::add_line_info(cx, instr);
}

fn GEP(cx: @block_ctxt, Pointer: ValueRef, Indices: [ValueRef]) -> ValueRef {
    if cx.unreachable { ret llvm::LLVMGetUndef(T_ptr(T_nil())); }
    unsafe {
        let instr = llvm::LLVMBuildGEP(B(cx), Pointer, vec::to_ptr(Indices),
                                       vec::len(Indices), noname());
        //debuginfo::add_line_info(cx, instr);
        ret instr;
    }
}

// Simple wrapper around GEP that takes an array of ints and wraps them
// in C_i32()
fn GEPi(cx: @block_ctxt, base: ValueRef, ixs: [int]) -> ValueRef {
    let v: [ValueRef] = [];
    for i: int in ixs { v += [C_i32(i as i32)]; }
    ret InBoundsGEP(cx, base, v);
}

fn InBoundsGEP(cx: @block_ctxt, Pointer: ValueRef, Indices: [ValueRef]) ->
   ValueRef {
    if cx.unreachable { ret llvm::LLVMGetUndef(T_ptr(T_nil())); }
    unsafe {
        let instr = llvm::LLVMBuildInBoundsGEP(B(cx), Pointer,
                                               vec::to_ptr(Indices),
                                               vec::len(Indices), noname());
        //debuginfo::add_line_info(cx, instr);
        ret instr;
    }
}

fn StructGEP(cx: @block_ctxt, Pointer: ValueRef, Idx: uint) -> ValueRef {
    if cx.unreachable { ret llvm::LLVMGetUndef(T_ptr(T_nil())); }
    let instr = llvm::LLVMBuildStructGEP(B(cx), Pointer, Idx, noname());
    //debuginfo::add_line_info(cx, instr);
    ret instr;
}

fn GlobalString(cx: @block_ctxt, _Str: sbuf) -> ValueRef {
    if cx.unreachable { ret llvm::LLVMGetUndef(T_ptr(T_i8())); }
    let instr = llvm::LLVMBuildGlobalString(B(cx), _Str, noname());
    debuginfo::add_line_info(cx, instr);
    ret instr;
}

fn GlobalStringPtr(cx: @block_ctxt, _Str: sbuf) -> ValueRef {
    if cx.unreachable { ret llvm::LLVMGetUndef(T_ptr(T_i8())); }
    let instr = llvm::LLVMBuildGlobalStringPtr(B(cx), _Str, noname());
    debuginfo::add_line_info(cx, instr);
    ret instr;
}

/* Casts */
fn Trunc(cx: @block_ctxt, Val: ValueRef, DestTy: TypeRef) -> ValueRef {
    if cx.unreachable { ret llvm::LLVMGetUndef(DestTy); }
    let instr = llvm::LLVMBuildTrunc(B(cx), Val, DestTy, noname());
    //debuginfo::add_line_info(cx, instr);
    ret instr;
}

fn ZExt(cx: @block_ctxt, Val: ValueRef, DestTy: TypeRef) -> ValueRef {
    if cx.unreachable { ret llvm::LLVMGetUndef(DestTy); }
    let instr = llvm::LLVMBuildZExt(B(cx), Val, DestTy, noname());
    //debuginfo::add_line_info(cx, instr);
    ret instr;
}

fn SExt(cx: @block_ctxt, Val: ValueRef, DestTy: TypeRef) -> ValueRef {
    if cx.unreachable { ret llvm::LLVMGetUndef(DestTy); }
    let instr = llvm::LLVMBuildSExt(B(cx), Val, DestTy, noname());
    //debuginfo::add_line_info(cx, instr);
    ret instr;
}

fn FPToUI(cx: @block_ctxt, Val: ValueRef, DestTy: TypeRef) -> ValueRef {
    if cx.unreachable { ret llvm::LLVMGetUndef(DestTy); }
    let instr = llvm::LLVMBuildFPToUI(B(cx), Val, DestTy, noname());
    //debuginfo::add_line_info(cx, instr);
    ret instr;
}

fn FPToSI(cx: @block_ctxt, Val: ValueRef, DestTy: TypeRef) -> ValueRef {
    if cx.unreachable { ret llvm::LLVMGetUndef(DestTy); }
    let instr = llvm::LLVMBuildFPToSI(B(cx), Val, DestTy, noname());
    //debuginfo::add_line_info(cx, instr);
    ret instr;
}

fn UIToFP(cx: @block_ctxt, Val: ValueRef, DestTy: TypeRef) -> ValueRef {
    if cx.unreachable { ret llvm::LLVMGetUndef(DestTy); }
    let instr = llvm::LLVMBuildUIToFP(B(cx), Val, DestTy, noname());
    //debuginfo::add_line_info(cx, instr);
    ret instr;
}

fn SIToFP(cx: @block_ctxt, Val: ValueRef, DestTy: TypeRef) -> ValueRef {
    if cx.unreachable { ret llvm::LLVMGetUndef(DestTy); }
    let instr = llvm::LLVMBuildSIToFP(B(cx), Val, DestTy, noname());
    //debuginfo::add_line_info(cx, instr);
    ret instr;
}

fn FPTrunc(cx: @block_ctxt, Val: ValueRef, DestTy: TypeRef) -> ValueRef {
    if cx.unreachable { ret llvm::LLVMGetUndef(DestTy); }
    let instr = llvm::LLVMBuildFPTrunc(B(cx), Val, DestTy, noname());
    //debuginfo::add_line_info(cx, instr);
    ret instr;
}

fn FPExt(cx: @block_ctxt, Val: ValueRef, DestTy: TypeRef) -> ValueRef {
    if cx.unreachable { ret llvm::LLVMGetUndef(DestTy); }
    let instr = llvm::LLVMBuildFPExt(B(cx), Val, DestTy, noname());
    //debuginfo::add_line_info(cx, instr);
    ret instr;
}

fn PtrToInt(cx: @block_ctxt, Val: ValueRef, DestTy: TypeRef) -> ValueRef {
    if cx.unreachable { ret llvm::LLVMGetUndef(DestTy); }
    let instr = llvm::LLVMBuildPtrToInt(B(cx), Val, DestTy, noname());
    //debuginfo::add_line_info(cx, instr);
    ret instr;
}

fn IntToPtr(cx: @block_ctxt, Val: ValueRef, DestTy: TypeRef) -> ValueRef {
    if cx.unreachable { ret llvm::LLVMGetUndef(DestTy); }
    let instr = llvm::LLVMBuildIntToPtr(B(cx), Val, DestTy, noname());
    //debuginfo::add_line_info(cx, instr);
    ret instr;
}

fn BitCast(cx: @block_ctxt, Val: ValueRef, DestTy: TypeRef) -> ValueRef {
    if cx.unreachable { ret llvm::LLVMGetUndef(DestTy); }
    let instr = llvm::LLVMBuildBitCast(B(cx), Val, DestTy, noname());
    //debuginfo::add_line_info(cx, instr);
    ret instr;
}

fn ZExtOrBitCast(cx: @block_ctxt, Val: ValueRef, DestTy: TypeRef) ->
   ValueRef {
    if cx.unreachable { ret llvm::LLVMGetUndef(DestTy); }
    let instr = llvm::LLVMBuildZExtOrBitCast(B(cx), Val, DestTy, noname());
    //debuginfo::add_line_info(cx, instr);
    ret instr;
}

fn SExtOrBitCast(cx: @block_ctxt, Val: ValueRef, DestTy: TypeRef) ->
   ValueRef {
    if cx.unreachable { ret llvm::LLVMGetUndef(DestTy); }
    let instr = llvm::LLVMBuildSExtOrBitCast(B(cx), Val, DestTy, noname());
    //debuginfo::add_line_info(cx, instr);
    ret instr;
}

fn TruncOrBitCast(cx: @block_ctxt, Val: ValueRef, DestTy: TypeRef) ->
   ValueRef {
    if cx.unreachable { ret llvm::LLVMGetUndef(DestTy); }
    let instr = llvm::LLVMBuildTruncOrBitCast(B(cx), Val, DestTy, noname());
    //debuginfo::add_line_info(cx, instr);
    ret instr;
}

fn Cast(cx: @block_ctxt, Op: Opcode, Val: ValueRef, DestTy: TypeRef,
        _Name: sbuf) -> ValueRef {
    if cx.unreachable { ret llvm::LLVMGetUndef(DestTy); }
    let instr = llvm::LLVMBuildCast(B(cx), Op, Val, DestTy, noname());
    //debuginfo::add_line_info(cx, instr);
    ret instr;
}

fn PointerCast(cx: @block_ctxt, Val: ValueRef, DestTy: TypeRef) -> ValueRef {
    if cx.unreachable { ret llvm::LLVMGetUndef(DestTy); }
    let instr = llvm::LLVMBuildPointerCast(B(cx), Val, DestTy, noname());
    //debuginfo::add_line_info(cx, instr);
    ret instr;
}

fn IntCast(cx: @block_ctxt, Val: ValueRef, DestTy: TypeRef) -> ValueRef {
    if cx.unreachable { ret llvm::LLVMGetUndef(DestTy); }
    let instr = llvm::LLVMBuildIntCast(B(cx), Val, DestTy, noname());
    //debuginfo::add_line_info(cx, instr);
    ret instr;
}

fn FPCast(cx: @block_ctxt, Val: ValueRef, DestTy: TypeRef) -> ValueRef {
    if cx.unreachable { ret llvm::LLVMGetUndef(DestTy); }
    let instr = llvm::LLVMBuildFPCast(B(cx), Val, DestTy, noname());
    //debuginfo::add_line_info(cx, instr);
    ret instr;
}


/* Comparisons */
fn ICmp(cx: @block_ctxt, Op: uint, LHS: ValueRef, RHS: ValueRef) -> ValueRef {
    if cx.unreachable { ret llvm::LLVMGetUndef(T_i1()); }
    let instr = llvm::LLVMBuildICmp(B(cx), Op, LHS, RHS, noname());
    debuginfo::add_line_info(cx, instr);
    ret instr;
}

fn FCmp(cx: @block_ctxt, Op: uint, LHS: ValueRef, RHS: ValueRef) -> ValueRef {
    if cx.unreachable { ret llvm::LLVMGetUndef(T_i1()); }
    let instr = llvm::LLVMBuildFCmp(B(cx), Op, LHS, RHS, noname());
    debuginfo::add_line_info(cx, instr);
    ret instr;
}

/* Miscellaneous instructions */
fn EmptyPhi(cx: @block_ctxt, Ty: TypeRef) -> ValueRef {
    if cx.unreachable { ret llvm::LLVMGetUndef(Ty); }
    let instr = llvm::LLVMBuildPhi(B(cx), Ty, noname());
    debuginfo::add_line_info(cx, instr);
    ret instr;
}

fn Phi(cx: @block_ctxt, Ty: TypeRef, vals: [ValueRef], bbs: [BasicBlockRef])
   -> ValueRef {
    if cx.unreachable { ret llvm::LLVMGetUndef(Ty); }
    assert (vec::len::<ValueRef>(vals) == vec::len::<BasicBlockRef>(bbs));
    let phi = EmptyPhi(cx, Ty);
    unsafe {
        llvm::LLVMAddIncoming(phi, vec::to_ptr(vals), vec::to_ptr(bbs),
                              vec::len(vals));
        ret phi;
    }
}

fn AddIncomingToPhi(phi: ValueRef, val: ValueRef, bb: BasicBlockRef) {
    if llvm::LLVMIsUndef(phi) == lib::llvm::True { ret; }
    unsafe {
        let valptr = unsafe::reinterpret_cast(ptr::addr_of(val));
        let bbptr = unsafe::reinterpret_cast(ptr::addr_of(bb));
        llvm::LLVMAddIncoming(phi, valptr, bbptr, 1u);
    }
}

fn _UndefReturn(cx: @block_ctxt, Fn: ValueRef) -> ValueRef {
    let ccx = cx.fcx.lcx.ccx;
    let ty = val_ty(Fn);
    let retty = if llvm::LLVMGetTypeKind(ty) == 8 {
        llvm::LLVMGetReturnType(ty) } else { ccx.int_type };
    ret llvm::LLVMGetUndef(retty);
}

fn add_span_comment(bcx: @block_ctxt, sp: span, text: str) {
    let ccx = bcx_ccx(bcx);
    if (!ccx.sess.get_opts().no_asm_comments) {
        add_comment(bcx, text + " (" + ccx.sess.span_str(sp) + ")");
    }
}

fn add_comment(bcx: @block_ctxt, text: str) {
    let ccx = bcx_ccx(bcx);
    if (!ccx.sess.get_opts().no_asm_comments) {
        check str::is_not_empty("$");
        let sanitized = str::replace(text, "$", "");
        let comment_text = "; " + sanitized;
        let asm = str::as_buf(comment_text, { |c|
            str::as_buf("", { |e|
                llvm::LLVMConstInlineAsm(T_fn([], T_void()), c, e, 0, 0)})});
        Call(bcx, asm, []);
    }
}

fn Call(cx: @block_ctxt, Fn: ValueRef, Args: [ValueRef]) -> ValueRef {
    if cx.unreachable { ret _UndefReturn(cx, Fn); }
    unsafe {
        let instr = llvm::LLVMBuildCall(B(cx), Fn, vec::to_ptr(Args),
                                        vec::len(Args), noname());
        debuginfo::add_line_info(cx, instr);
        ret instr;
    }
}

fn FastCall(cx: @block_ctxt, Fn: ValueRef, Args: [ValueRef]) -> ValueRef {
    if cx.unreachable { ret _UndefReturn(cx, Fn); }
    unsafe {
        let v = llvm::LLVMBuildCall(B(cx), Fn, vec::to_ptr(Args),
                                    vec::len(Args), noname());
        llvm::LLVMSetInstructionCallConv(v, lib::llvm::LLVMFastCallConv);
        debuginfo::add_line_info(cx, v);
        ret v;
    }
}

fn CallWithConv(cx: @block_ctxt, Fn: ValueRef, Args: [ValueRef], Conv: uint)
   -> ValueRef {
    if cx.unreachable { ret _UndefReturn(cx, Fn); }
    unsafe {
        let v = llvm::LLVMBuildCall(B(cx), Fn, vec::to_ptr(Args),
                                    vec::len(Args), noname());
        llvm::LLVMSetInstructionCallConv(v, Conv);
        debuginfo::add_line_info(cx, v);
        ret v;
    }
}

fn Select(cx: @block_ctxt, If: ValueRef, Then: ValueRef, Else: ValueRef) ->
   ValueRef {
    if cx.unreachable { ret _Undef(Then); }
    let instr = llvm::LLVMBuildSelect(B(cx), If, Then, Else, noname());
    debuginfo::add_line_info(cx, instr);
    ret instr;
}

fn VAArg(cx: @block_ctxt, list: ValueRef, Ty: TypeRef) -> ValueRef {
    if cx.unreachable { ret llvm::LLVMGetUndef(Ty); }
    let instr = llvm::LLVMBuildVAArg(B(cx), list, Ty, noname());
    debuginfo::add_line_info(cx, instr);
    ret instr;
}

fn ExtractElement(cx: @block_ctxt, VecVal: ValueRef, Index: ValueRef) ->
   ValueRef {
    if cx.unreachable { ret llvm::LLVMGetUndef(T_nil()); }
    let instr = llvm::LLVMBuildExtractElement(B(cx), VecVal, Index, noname());
    debuginfo::add_line_info(cx, instr);
    ret instr;
}

fn InsertElement(cx: @block_ctxt, VecVal: ValueRef, EltVal: ValueRef,
                 Index: ValueRef) {
    if cx.unreachable { ret; }
    let instr = llvm::LLVMBuildInsertElement(B(cx), VecVal, EltVal, Index,
                                             noname());
    debuginfo::add_line_info(cx, instr);
}

fn ShuffleVector(cx: @block_ctxt, V1: ValueRef, V2: ValueRef,
                 Mask: ValueRef) {
    if cx.unreachable { ret; }
    let instr = llvm::LLVMBuildShuffleVector(B(cx), V1, V2, Mask, noname());
    debuginfo::add_line_info(cx, instr);
}

fn ExtractValue(cx: @block_ctxt, AggVal: ValueRef, Index: uint) -> ValueRef {
    if cx.unreachable { ret llvm::LLVMGetUndef(T_nil()); }
    let instr = llvm::LLVMBuildExtractValue(B(cx), AggVal, Index, noname());
    debuginfo::add_line_info(cx, instr);
    ret instr;
}

fn InsertValue(cx: @block_ctxt, AggVal: ValueRef, EltVal: ValueRef,
               Index: uint) {
    if cx.unreachable { ret; }
    let instr = llvm::LLVMBuildInsertValue(B(cx), AggVal, EltVal, Index,
                                           noname());
    debuginfo::add_line_info(cx, instr);
}

fn IsNull(cx: @block_ctxt, Val: ValueRef) -> ValueRef {
    if cx.unreachable { ret llvm::LLVMGetUndef(T_i1()); }
    let instr = llvm::LLVMBuildIsNull(B(cx), Val, noname());
    debuginfo::add_line_info(cx, instr);
    ret instr;
}

fn IsNotNull(cx: @block_ctxt, Val: ValueRef) -> ValueRef {
    if cx.unreachable { ret llvm::LLVMGetUndef(T_i1()); }
    let instr = llvm::LLVMBuildIsNotNull(B(cx), Val, noname());
    debuginfo::add_line_info(cx, instr);
    ret instr;
}

fn PtrDiff(cx: @block_ctxt, LHS: ValueRef, RHS: ValueRef) -> ValueRef {
    let ccx = cx.fcx.lcx.ccx;
    if cx.unreachable { ret llvm::LLVMGetUndef(ccx.int_type); }
    let instr = llvm::LLVMBuildPtrDiff(B(cx), LHS, RHS, noname());
    debuginfo::add_line_info(cx, instr);
    ret instr;
}

fn Trap(cx: @block_ctxt) {
    if cx.unreachable { ret; }
    let b = B(cx);
    let BB: BasicBlockRef = llvm::LLVMGetInsertBlock(b);
    let FN: ValueRef = llvm::LLVMGetBasicBlockParent(BB);
    let M: ModuleRef = llvm::LLVMGetGlobalParent(FN);
    let T: ValueRef = str::as_buf("llvm.trap", {|buf|
        llvm::LLVMGetNamedFunction(M, buf)
    });
    assert (T as int != 0);
    let Args: [ValueRef] = [];
    unsafe {
        let instr = llvm::LLVMBuildCall(b, T, vec::to_ptr(Args),
                                        vec::len(Args), noname());
        debuginfo::add_line_info(cx, instr);
    }
}

fn LandingPad(cx: @block_ctxt, Ty: TypeRef, PersFn: ValueRef,
              NumClauses: uint) -> ValueRef {
    assert !cx.terminated && !cx.unreachable;
    let instr = llvm::LLVMBuildLandingPad(B(cx), Ty, PersFn, NumClauses,
                                          noname());
    debuginfo::add_line_info(cx, instr);
    ret instr;
}

fn SetCleanup(_cx: @block_ctxt, LandingPad: ValueRef) {
    llvm::LLVMSetCleanup(LandingPad, lib::llvm::True);
}

fn Resume(cx: @block_ctxt, Exn: ValueRef) -> ValueRef {
    assert (!cx.terminated);
    cx.terminated = true;
    let instr = llvm::LLVMBuildResume(B(cx), Exn);
    debuginfo::add_line_info(cx, instr);
    ret instr;
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
