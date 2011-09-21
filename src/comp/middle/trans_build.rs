import std::{vec, str};
import std::str::sbuf;
import lib::llvm::llvm;
import llvm::{ValueRef, TypeRef, BasicBlockRef, BuilderRef, Opcode,
              ModuleRef};
import trans_common::{block_ctxt, T_ptr, T_nil, T_int, T_i8, T_i1, val_ty};

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
    llvm::LLVMBuildRetVoid(B(cx));
}

fn Ret(cx: @block_ctxt, V: ValueRef) {
    if cx.unreachable { ret; }
    assert (!cx.terminated);
    cx.terminated = true;
    llvm::LLVMBuildRet(B(cx), V);
}

fn AggregateRet(cx: @block_ctxt, RetVals: [ValueRef]) {
    if cx.unreachable { ret; }
    assert (!cx.terminated);
    cx.terminated = true;
    llvm::LLVMBuildAggregateRet(B(cx), vec::to_ptr(RetVals),
                                vec::len(RetVals));
}

fn Br(cx: @block_ctxt, Dest: BasicBlockRef) {
    if cx.unreachable { ret; }
    assert (!cx.terminated);
    cx.terminated = true;
    llvm::LLVMBuildBr(B(cx), Dest);
}

fn CondBr(cx: @block_ctxt, If: ValueRef, Then: BasicBlockRef,
          Else: BasicBlockRef) {
    if cx.unreachable { ret; }
    assert (!cx.terminated);
    cx.terminated = true;
    llvm::LLVMBuildCondBr(B(cx), If, Then, Else);
}

fn Switch(cx: @block_ctxt, V: ValueRef, Else: BasicBlockRef, NumCases: uint)
    -> ValueRef {
    if cx.unreachable { ret _Undef(V); }
    assert !cx.terminated;
    cx.terminated = true;
    ret llvm::LLVMBuildSwitch(B(cx), V, Else, NumCases);
}

fn AddCase(S: ValueRef, OnVal: ValueRef, Dest: BasicBlockRef) {
    if llvm::LLVMIsUndef(S) == lib::llvm::True { ret; }
    llvm::LLVMAddCase(S, OnVal, Dest);
}

fn IndirectBr(cx: @block_ctxt, Addr: ValueRef, NumDests: uint) {
    if cx.unreachable { ret; }
    assert (!cx.terminated);
    cx.terminated = true;
    llvm::LLVMBuildIndirectBr(B(cx), Addr, NumDests);
}

fn Invoke(cx: @block_ctxt, Fn: ValueRef, Args: [ValueRef],
          Then: BasicBlockRef, Catch: BasicBlockRef) {
    if cx.unreachable { ret; }
    assert (!cx.terminated);
    cx.terminated = true;
    str::as_buf("", {|buf|
        llvm::LLVMBuildInvoke(B(cx), Fn, vec::to_ptr(Args),
                              vec::len(Args), Then, Catch, buf)
    });
}

fn FastInvoke(cx: @block_ctxt, Fn: ValueRef, Args: [ValueRef],
              Then: BasicBlockRef, Catch: BasicBlockRef) {
    if cx.unreachable { ret; }
    assert (!cx.terminated);
    cx.terminated = true;
    let v = str::as_buf("", {|buf|
        llvm::LLVMBuildInvoke(B(cx), Fn, vec::to_ptr(Args),
                              vec::len(Args), Then, Catch, buf)
    });
    llvm::LLVMSetInstructionCallConv(v, lib::llvm::LLVMFastCallConv);
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
    ret str::as_buf("", {|buf| llvm::LLVMBuildAdd(B(cx), LHS, RHS, buf) });
}

fn NSWAdd(cx: @block_ctxt, LHS: ValueRef, RHS: ValueRef) -> ValueRef {
    if cx.unreachable { ret _Undef(LHS); }
    ret str::as_buf("", {|buf| llvm::LLVMBuildNSWAdd(B(cx), LHS, RHS, buf) });
}

fn NUWAdd(cx: @block_ctxt, LHS: ValueRef, RHS: ValueRef) -> ValueRef {
    if cx.unreachable { ret _Undef(LHS); }
    ret str::as_buf("", {|buf| llvm::LLVMBuildNUWAdd(B(cx), LHS, RHS, buf) });
}

fn FAdd(cx: @block_ctxt, LHS: ValueRef, RHS: ValueRef) -> ValueRef {
    if cx.unreachable { ret _Undef(LHS); }
    ret str::as_buf("", {|buf| llvm::LLVMBuildFAdd(B(cx), LHS, RHS, buf) });
}

fn Sub(cx: @block_ctxt, LHS: ValueRef, RHS: ValueRef) -> ValueRef {
    if cx.unreachable { ret _Undef(LHS); }
    ret str::as_buf("", {|buf| llvm::LLVMBuildSub(B(cx), LHS, RHS, buf) });
}

fn NSWSub(cx: @block_ctxt, LHS: ValueRef, RHS: ValueRef) -> ValueRef {
    if cx.unreachable { ret _Undef(LHS); }
    ret str::as_buf("", {|buf| llvm::LLVMBuildNSWSub(B(cx), LHS, RHS, buf) });
}

fn NUWSub(cx: @block_ctxt, LHS: ValueRef, RHS: ValueRef) -> ValueRef {
    if cx.unreachable { ret _Undef(LHS); }
    ret str::as_buf("", {|buf| llvm::LLVMBuildNUWSub(B(cx), LHS, RHS, buf) });
}

fn FSub(cx: @block_ctxt, LHS: ValueRef, RHS: ValueRef) -> ValueRef {
    if cx.unreachable { ret _Undef(LHS); }
    ret str::as_buf("", {|buf| llvm::LLVMBuildFSub(B(cx), LHS, RHS, buf) });
}

fn Mul(cx: @block_ctxt, LHS: ValueRef, RHS: ValueRef) -> ValueRef {
    if cx.unreachable { ret _Undef(LHS); }
    ret str::as_buf("", {|buf| llvm::LLVMBuildMul(B(cx), LHS, RHS, buf) });
}

fn NSWMul(cx: @block_ctxt, LHS: ValueRef, RHS: ValueRef) -> ValueRef {
    if cx.unreachable { ret _Undef(LHS); }
    ret str::as_buf("", {|buf| llvm::LLVMBuildNSWMul(B(cx), LHS, RHS, buf) });
}

fn NUWMul(cx: @block_ctxt, LHS: ValueRef, RHS: ValueRef) -> ValueRef {
    if cx.unreachable { ret _Undef(LHS); }
    ret str::as_buf("", {|buf| llvm::LLVMBuildNUWMul(B(cx), LHS, RHS, buf) });
}

fn FMul(cx: @block_ctxt, LHS: ValueRef, RHS: ValueRef) -> ValueRef {
    if cx.unreachable { ret _Undef(LHS); }
    ret str::as_buf("", {|buf| llvm::LLVMBuildFMul(B(cx), LHS, RHS, buf) });
}

fn UDiv(cx: @block_ctxt, LHS: ValueRef, RHS: ValueRef) -> ValueRef {
    if cx.unreachable { ret _Undef(LHS); }
    ret str::as_buf("", {|buf| llvm::LLVMBuildUDiv(B(cx), LHS, RHS, buf) });
}

fn SDiv(cx: @block_ctxt, LHS: ValueRef, RHS: ValueRef) -> ValueRef {
    if cx.unreachable { ret _Undef(LHS); }
    ret str::as_buf("", {|buf| llvm::LLVMBuildSDiv(B(cx), LHS, RHS, buf) });
}

fn ExactSDiv(cx: @block_ctxt, LHS: ValueRef, RHS: ValueRef) -> ValueRef {
    if cx.unreachable { ret _Undef(LHS); }
    ret str::as_buf("",
                    {|buf| llvm::LLVMBuildExactSDiv(B(cx), LHS, RHS, buf) });
}

fn FDiv(cx: @block_ctxt, LHS: ValueRef, RHS: ValueRef) -> ValueRef {
    if cx.unreachable { ret _Undef(LHS); }
    ret str::as_buf("", {|buf| llvm::LLVMBuildFDiv(B(cx), LHS, RHS, buf) });
}

fn URem(cx: @block_ctxt, LHS: ValueRef, RHS: ValueRef) -> ValueRef {
    if cx.unreachable { ret _Undef(LHS); }
    ret str::as_buf("", {|buf| llvm::LLVMBuildURem(B(cx), LHS, RHS, buf) });
}

fn SRem(cx: @block_ctxt, LHS: ValueRef, RHS: ValueRef) -> ValueRef {
    if cx.unreachable { ret _Undef(LHS); }
    ret str::as_buf("", {|buf| llvm::LLVMBuildSRem(B(cx), LHS, RHS, buf) });
}

fn FRem(cx: @block_ctxt, LHS: ValueRef, RHS: ValueRef) -> ValueRef {
    if cx.unreachable { ret _Undef(LHS); }
    ret str::as_buf("", {|buf| llvm::LLVMBuildFRem(B(cx), LHS, RHS, buf) });
}

fn Shl(cx: @block_ctxt, LHS: ValueRef, RHS: ValueRef) -> ValueRef {
    if cx.unreachable { ret _Undef(LHS); }
    ret str::as_buf("", {|buf| llvm::LLVMBuildShl(B(cx), LHS, RHS, buf) });
}

fn LShr(cx: @block_ctxt, LHS: ValueRef, RHS: ValueRef) -> ValueRef {
    if cx.unreachable { ret _Undef(LHS); }
    ret str::as_buf("", {|buf| llvm::LLVMBuildLShr(B(cx), LHS, RHS, buf) });
}

fn AShr(cx: @block_ctxt, LHS: ValueRef, RHS: ValueRef) -> ValueRef {
    if cx.unreachable { ret _Undef(LHS); }
    ret str::as_buf("", {|buf| llvm::LLVMBuildAShr(B(cx), LHS, RHS, buf) });
}

fn And(cx: @block_ctxt, LHS: ValueRef, RHS: ValueRef) -> ValueRef {
    if cx.unreachable { ret _Undef(LHS); }
    ret str::as_buf("", {|buf| llvm::LLVMBuildAnd(B(cx), LHS, RHS, buf) });
}

fn Or(cx: @block_ctxt, LHS: ValueRef, RHS: ValueRef) -> ValueRef {
    if cx.unreachable { ret _Undef(LHS); }
    ret str::as_buf("", {|buf| llvm::LLVMBuildOr(B(cx), LHS, RHS, buf) });
}

fn Xor(cx: @block_ctxt, LHS: ValueRef, RHS: ValueRef) -> ValueRef {
    if cx.unreachable { ret _Undef(LHS); }
    ret str::as_buf("", {|buf| llvm::LLVMBuildXor(B(cx), LHS, RHS, buf) });
}

fn BinOp(cx: @block_ctxt, Op: Opcode, LHS: ValueRef, RHS: ValueRef) ->
   ValueRef {
    if cx.unreachable { ret _Undef(LHS); }
    ret str::as_buf("",
                    {|buf| llvm::LLVMBuildBinOp(B(cx), Op, LHS, RHS, buf) });
}

fn Neg(cx: @block_ctxt, V: ValueRef) -> ValueRef {
    if cx.unreachable { ret _Undef(V); }
    ret str::as_buf("", {|buf| llvm::LLVMBuildNeg(B(cx), V, buf) });
}

fn NSWNeg(cx: @block_ctxt, V: ValueRef) -> ValueRef {
    if cx.unreachable { ret _Undef(V); }
    ret str::as_buf("", {|buf| llvm::LLVMBuildNSWNeg(B(cx), V, buf) });
}

fn NUWNeg(cx: @block_ctxt, V: ValueRef) -> ValueRef {
    if cx.unreachable { ret _Undef(V); }
    ret str::as_buf("", {|buf| llvm::LLVMBuildNUWNeg(B(cx), V, buf) });
}
fn FNeg(cx: @block_ctxt, V: ValueRef) -> ValueRef {
    if cx.unreachable { ret _Undef(V); }
    ret str::as_buf("", {|buf| llvm::LLVMBuildFNeg(B(cx), V, buf) });
}

fn Not(cx: @block_ctxt, V: ValueRef) -> ValueRef {
    if cx.unreachable { ret _Undef(V); }
    ret str::as_buf("", {|buf| llvm::LLVMBuildNot(B(cx), V, buf) });
}

/* Memory */
fn Malloc(cx: @block_ctxt, Ty: TypeRef) -> ValueRef {
    if cx.unreachable { ret llvm::LLVMGetUndef(T_ptr(T_i8())); }
    ret str::as_buf("", {|buf| llvm::LLVMBuildMalloc(B(cx), Ty, buf) });
}

fn ArrayMalloc(cx: @block_ctxt, Ty: TypeRef, Val: ValueRef) -> ValueRef {
    if cx.unreachable { ret llvm::LLVMGetUndef(T_ptr(T_i8())); }
    ret str::as_buf("",
                    {|buf| llvm::LLVMBuildArrayMalloc(B(cx), Ty, Val, buf) });
}

fn Alloca(cx: @block_ctxt, Ty: TypeRef) -> ValueRef {
    if cx.unreachable { ret llvm::LLVMGetUndef(T_ptr(Ty)); }
    ret str::as_buf("", {|buf| llvm::LLVMBuildAlloca(B(cx), Ty, buf) });
}

fn ArrayAlloca(cx: @block_ctxt, Ty: TypeRef, Val: ValueRef) -> ValueRef {
    if cx.unreachable { ret llvm::LLVMGetUndef(T_ptr(Ty)); }
    ret str::as_buf("",
                    {|buf| llvm::LLVMBuildArrayAlloca(B(cx), Ty, Val, buf) });
}

fn Free(cx: @block_ctxt, PointerVal: ValueRef) {
    if cx.unreachable { ret; }
    llvm::LLVMBuildFree(B(cx), PointerVal);
}

fn Load(cx: @block_ctxt, PointerVal: ValueRef) -> ValueRef {
    if cx.unreachable {
        let ty = val_ty(PointerVal);
        let eltty = if llvm::LLVMGetTypeKind(ty) == 11 {
            llvm::LLVMGetElementType(ty) } else { T_int() };
        ret llvm::LLVMGetUndef(eltty);
    }
    ret str::as_buf("", {|buf| llvm::LLVMBuildLoad(B(cx), PointerVal, buf) });
}

fn Store(cx: @block_ctxt, Val: ValueRef, Ptr: ValueRef) {
    if cx.unreachable { ret; }
    llvm::LLVMBuildStore(B(cx), Val, Ptr);
}

fn GEP(cx: @block_ctxt, Pointer: ValueRef, Indices: [ValueRef]) -> ValueRef {
    if cx.unreachable { ret llvm::LLVMGetUndef(T_ptr(T_nil())); }
    ret str::as_buf("", {|buf|
        llvm::LLVMBuildGEP(B(cx), Pointer,
                           vec::to_ptr(Indices),
                           vec::len(Indices), buf)
    });
}

fn InBoundsGEP(cx: @block_ctxt, Pointer: ValueRef, Indices: [ValueRef]) ->
   ValueRef {
    if cx.unreachable { ret llvm::LLVMGetUndef(T_ptr(T_nil())); }
    ret str::as_buf("", {|buf|
        llvm::LLVMBuildInBoundsGEP(B(cx), Pointer, vec::to_ptr(Indices),
                                   vec::len(Indices), buf)
    });
}

fn StructGEP(cx: @block_ctxt, Pointer: ValueRef, Idx: uint) -> ValueRef {
    if cx.unreachable { ret llvm::LLVMGetUndef(T_ptr(T_nil())); }
    ret str::as_buf("",
                    {|buf|
                        llvm::LLVMBuildStructGEP(B(cx), Pointer, Idx, buf)
                    });
}

fn GlobalString(cx: @block_ctxt, _Str: sbuf) -> ValueRef {
    if cx.unreachable { ret llvm::LLVMGetUndef(T_ptr(T_i8())); }
    ret str::as_buf("",
                    {|buf| llvm::LLVMBuildGlobalString(B(cx), _Str, buf) });
}

fn GlobalStringPtr(cx: @block_ctxt, _Str: sbuf) -> ValueRef {
    if cx.unreachable { ret llvm::LLVMGetUndef(T_ptr(T_i8())); }
    ret str::as_buf("",
                    {|buf|
                        llvm::LLVMBuildGlobalStringPtr(B(cx), _Str, buf)
                    });
}

/* Casts */
fn Trunc(cx: @block_ctxt, Val: ValueRef, DestTy: TypeRef) -> ValueRef {
    if cx.unreachable { ret llvm::LLVMGetUndef(DestTy); }
    ret str::as_buf("",
                    {|buf| llvm::LLVMBuildTrunc(B(cx), Val, DestTy, buf) });
}

fn ZExt(cx: @block_ctxt, Val: ValueRef, DestTy: TypeRef) -> ValueRef {
    if cx.unreachable { ret llvm::LLVMGetUndef(DestTy); }
    ret str::as_buf("",
                    {|buf| llvm::LLVMBuildZExt(B(cx), Val, DestTy, buf) });
}

fn SExt(cx: @block_ctxt, Val: ValueRef, DestTy: TypeRef) -> ValueRef {
    if cx.unreachable { ret llvm::LLVMGetUndef(DestTy); }
    ret str::as_buf("",
                    {|buf| llvm::LLVMBuildSExt(B(cx), Val, DestTy, buf) });
}

fn FPToUI(cx: @block_ctxt, Val: ValueRef, DestTy: TypeRef) -> ValueRef {
    if cx.unreachable { ret llvm::LLVMGetUndef(DestTy); }
    ret str::as_buf("",
                    {|buf| llvm::LLVMBuildFPToUI(B(cx), Val, DestTy, buf) });
}

fn FPToSI(cx: @block_ctxt, Val: ValueRef, DestTy: TypeRef) -> ValueRef {
    if cx.unreachable { ret llvm::LLVMGetUndef(DestTy); }
    ret str::as_buf("",
                    {|buf| llvm::LLVMBuildFPToSI(B(cx), Val, DestTy, buf) });
}

fn UIToFP(cx: @block_ctxt, Val: ValueRef, DestTy: TypeRef) -> ValueRef {
    if cx.unreachable { ret llvm::LLVMGetUndef(DestTy); }
    ret str::as_buf("",
                    {|buf| llvm::LLVMBuildUIToFP(B(cx), Val, DestTy, buf) });
}

fn SIToFP(cx: @block_ctxt, Val: ValueRef, DestTy: TypeRef) -> ValueRef {
    if cx.unreachable { ret llvm::LLVMGetUndef(DestTy); }
    ret str::as_buf("",
                    {|buf| llvm::LLVMBuildSIToFP(B(cx), Val, DestTy, buf) });
}

fn FPTrunc(cx: @block_ctxt, Val: ValueRef, DestTy: TypeRef) -> ValueRef {
    if cx.unreachable { ret llvm::LLVMGetUndef(DestTy); }
    ret str::as_buf("",
                    {|buf| llvm::LLVMBuildFPTrunc(B(cx), Val, DestTy, buf) });
}

fn FPExt(cx: @block_ctxt, Val: ValueRef, DestTy: TypeRef) -> ValueRef {
    if cx.unreachable { ret llvm::LLVMGetUndef(DestTy); }
    ret str::as_buf("",
                    {|buf| llvm::LLVMBuildFPExt(B(cx), Val, DestTy, buf) });
}

fn PtrToInt(cx: @block_ctxt, Val: ValueRef, DestTy: TypeRef) -> ValueRef {
    if cx.unreachable { ret llvm::LLVMGetUndef(DestTy); }
    ret str::as_buf("",
                    {|buf|
                        llvm::LLVMBuildPtrToInt(B(cx), Val, DestTy, buf)
                    });
}

fn IntToPtr(cx: @block_ctxt, Val: ValueRef, DestTy: TypeRef) -> ValueRef {
    if cx.unreachable { ret llvm::LLVMGetUndef(DestTy); }
    ret str::as_buf("",
                    {|buf|
                        llvm::LLVMBuildIntToPtr(B(cx), Val, DestTy, buf)
                    });
}

fn BitCast(cx: @block_ctxt, Val: ValueRef, DestTy: TypeRef) -> ValueRef {
    if cx.unreachable { ret llvm::LLVMGetUndef(DestTy); }
    ret str::as_buf("",
                    {|buf| llvm::LLVMBuildBitCast(B(cx), Val, DestTy, buf) });
}

fn ZExtOrBitCast(cx: @block_ctxt, Val: ValueRef, DestTy: TypeRef) ->
   ValueRef {
    if cx.unreachable { ret llvm::LLVMGetUndef(DestTy); }
    ret str::as_buf("",
                    {|buf|
                        llvm::LLVMBuildZExtOrBitCast(B(cx), Val, DestTy, buf)
                    });
}

fn SExtOrBitCast(cx: @block_ctxt, Val: ValueRef, DestTy: TypeRef) ->
   ValueRef {
    if cx.unreachable { ret llvm::LLVMGetUndef(DestTy); }
    ret str::as_buf("",
                    {|buf|
                        llvm::LLVMBuildSExtOrBitCast(B(cx), Val, DestTy, buf)
                    });
}

fn TruncOrBitCast(cx: @block_ctxt, Val: ValueRef, DestTy: TypeRef) ->
   ValueRef {
    if cx.unreachable { ret llvm::LLVMGetUndef(DestTy); }
    ret str::as_buf("",
                    {|buf|
                        llvm::LLVMBuildTruncOrBitCast(B(cx), Val, DestTy, buf)
                    });
}

fn Cast(cx: @block_ctxt, Op: Opcode, Val: ValueRef, DestTy: TypeRef,
        _Name: sbuf) -> ValueRef {
    if cx.unreachable { ret llvm::LLVMGetUndef(DestTy); }
    ret str::as_buf("",
                    {|buf|
                        llvm::LLVMBuildCast(B(cx), Op, Val, DestTy, buf)
                    });
}

fn PointerCast(cx: @block_ctxt, Val: ValueRef, DestTy: TypeRef) -> ValueRef {
    if cx.unreachable { ret llvm::LLVMGetUndef(DestTy); }
    ret str::as_buf("",
                    {|buf|
                        llvm::LLVMBuildPointerCast(B(cx), Val, DestTy, buf)
                    });
}

fn IntCast(cx: @block_ctxt, Val: ValueRef, DestTy: TypeRef) -> ValueRef {
    if cx.unreachable { ret llvm::LLVMGetUndef(DestTy); }
    ret str::as_buf("",
                    {|buf| llvm::LLVMBuildIntCast(B(cx), Val, DestTy, buf) });
}

fn FPCast(cx: @block_ctxt, Val: ValueRef, DestTy: TypeRef) -> ValueRef {
    if cx.unreachable { ret llvm::LLVMGetUndef(DestTy); }
    ret str::as_buf("",
                    {|buf| llvm::LLVMBuildFPCast(B(cx), Val, DestTy, buf) });
}


/* Comparisons */
fn ICmp(cx: @block_ctxt, Op: uint, LHS: ValueRef, RHS: ValueRef) -> ValueRef {
    if cx.unreachable { ret llvm::LLVMGetUndef(T_i1()); }
    ret str::as_buf("",
                    {|buf| llvm::LLVMBuildICmp(B(cx), Op, LHS, RHS, buf) });
}

fn FCmp(cx: @block_ctxt, Op: uint, LHS: ValueRef, RHS: ValueRef) -> ValueRef {
    if cx.unreachable { ret llvm::LLVMGetUndef(T_i1()); }
    ret str::as_buf("",
                    {|buf| llvm::LLVMBuildFCmp(B(cx), Op, LHS, RHS, buf) });
}


/* Miscellaneous instructions */
fn Phi(cx: @block_ctxt, Ty: TypeRef, vals: [ValueRef], bbs: [BasicBlockRef])
   -> ValueRef {
    if cx.unreachable { ret llvm::LLVMGetUndef(Ty); }
    let phi = str::as_buf("", {|buf| llvm::LLVMBuildPhi(B(cx), Ty, buf) });
    assert (vec::len::<ValueRef>(vals) == vec::len::<BasicBlockRef>(bbs));
    llvm::LLVMAddIncoming(phi, vec::to_ptr(vals), vec::to_ptr(bbs),
                          vec::len(vals));
    ret phi;
}

fn AddIncomingToPhi(phi: ValueRef, vals: [ValueRef], bbs: [BasicBlockRef]) {
    if llvm::LLVMIsUndef(phi) == lib::llvm::True { ret; }
    assert (vec::len::<ValueRef>(vals) == vec::len::<BasicBlockRef>(bbs));
    llvm::LLVMAddIncoming(phi, vec::to_ptr(vals), vec::to_ptr(bbs),
                          vec::len(vals));
}

fn _UndefReturn(Fn: ValueRef) -> ValueRef {
    let ty = val_ty(Fn);
    let retty = if llvm::LLVMGetTypeKind(ty) == 8 {
        llvm::LLVMGetReturnType(ty) } else { T_int() };
    ret llvm::LLVMGetUndef(retty);
}

fn Call(cx: @block_ctxt, Fn: ValueRef, Args: [ValueRef]) -> ValueRef {
    if cx.unreachable { ret _UndefReturn(Fn); }
    ret str::as_buf("",
                    {|buf|
                        llvm::LLVMBuildCall(B(cx), Fn, vec::to_ptr(Args),
                                            vec::len(Args), buf)
                    });
}

fn FastCall(cx: @block_ctxt, Fn: ValueRef, Args: [ValueRef]) -> ValueRef {
    if cx.unreachable { ret _UndefReturn(Fn); }
    let v =
        str::as_buf("",
                    {|buf|
                        llvm::LLVMBuildCall(B(cx), Fn, vec::to_ptr(Args),
                                            vec::len(Args), buf)
                    });
    llvm::LLVMSetInstructionCallConv(v, lib::llvm::LLVMFastCallConv);
    ret v;
}

fn CallWithConv(cx: @block_ctxt, Fn: ValueRef, Args: [ValueRef], Conv: uint)
   -> ValueRef {
    if cx.unreachable { ret _UndefReturn(Fn); }
    let v =
        str::as_buf("",
                    {|buf|
                        llvm::LLVMBuildCall(B(cx), Fn, vec::to_ptr(Args),
                                            vec::len(Args), buf)
                    });
    llvm::LLVMSetInstructionCallConv(v, Conv);
    ret v;
}

fn Select(cx: @block_ctxt, If: ValueRef, Then: ValueRef, Else: ValueRef) ->
   ValueRef {
    if cx.unreachable { ret _Undef(Then); }
    ret str::as_buf("",
                    {|buf|
                        llvm::LLVMBuildSelect(B(cx), If, Then, Else, buf)
                    });
}

fn VAArg(cx: @block_ctxt, list: ValueRef, Ty: TypeRef) -> ValueRef {
    if cx.unreachable { ret llvm::LLVMGetUndef(Ty); }
    ret str::as_buf("", {|buf| llvm::LLVMBuildVAArg(B(cx), list, Ty, buf) });
}

fn ExtractElement(cx: @block_ctxt, VecVal: ValueRef, Index: ValueRef) ->
   ValueRef {
    if cx.unreachable { ret llvm::LLVMGetUndef(T_nil()); }
    ret str::as_buf("",
                    {|buf|
                        llvm::LLVMBuildExtractElement(B(cx), VecVal, Index,
                                                      buf)
                    });
}

fn InsertElement(cx: @block_ctxt, VecVal: ValueRef, EltVal: ValueRef,
                 Index: ValueRef) {
    if cx.unreachable { ret; }
    str::as_buf("", {|buf|
        llvm::LLVMBuildInsertElement(B(cx), VecVal, EltVal, Index, buf)
    });
}

fn ShuffleVector(cx: @block_ctxt, V1: ValueRef, V2: ValueRef,
                 Mask: ValueRef) {
    if cx.unreachable { ret; }
    str::as_buf("", {|buf|
        llvm::LLVMBuildShuffleVector(B(cx), V1, V2, Mask, buf)
    });
}

fn ExtractValue(cx: @block_ctxt, AggVal: ValueRef, Index: uint) -> ValueRef {
    if cx.unreachable { ret llvm::LLVMGetUndef(T_nil()); }
    ret str::as_buf("",
                    {|buf|
                        llvm::LLVMBuildExtractValue(B(cx), AggVal, Index, buf)
                    });
}

fn InsertValue(cx: @block_ctxt, AggVal: ValueRef, EltVal: ValueRef,
               Index: uint) {
    if cx.unreachable { ret; }
    str::as_buf("", {|buf|
        llvm::LLVMBuildInsertValue(B(cx), AggVal, EltVal, Index, buf)
    });
}

fn IsNull(cx: @block_ctxt, Val: ValueRef) -> ValueRef {
    if cx.unreachable { ret llvm::LLVMGetUndef(T_i1()); }
    ret str::as_buf("", {|buf| llvm::LLVMBuildIsNull(B(cx), Val, buf) });
}

fn IsNotNull(cx: @block_ctxt, Val: ValueRef) -> ValueRef {
    if cx.unreachable { ret llvm::LLVMGetUndef(T_i1()); }
    ret str::as_buf("", {|buf| llvm::LLVMBuildIsNotNull(B(cx), Val, buf) });
}

fn PtrDiff(cx: @block_ctxt, LHS: ValueRef, RHS: ValueRef) -> ValueRef {
    if cx.unreachable { ret llvm::LLVMGetUndef(T_int()); }
    ret str::as_buf("",
                    {|buf| llvm::LLVMBuildPtrDiff(B(cx), LHS, RHS, buf) });
}

fn Trap(cx: @block_ctxt) {
    if cx.unreachable { ret; }
    let b = B(cx);
    let BB: BasicBlockRef = llvm::LLVMGetInsertBlock(b);
    let FN: ValueRef = llvm::LLVMGetBasicBlockParent(BB);
    let M: ModuleRef = llvm::LLVMGetGlobalParent(FN);
    let T: ValueRef =
        str::as_buf("llvm.trap", {|buf| llvm::LLVMGetNamedFunction(M, buf) });
    assert (T as int != 0);
    let Args: [ValueRef] = [];
    str::as_buf("", {|buf|
        llvm::LLVMBuildCall(b, T, vec::to_ptr(Args), vec::len(Args), buf)
    });
}

fn LandingPad(cx: @block_ctxt, Ty: TypeRef, PersFn: ValueRef,
              NumClauses: uint) -> ValueRef {
    assert !cx.terminated && !cx.unreachable;
    ret str::as_buf("", {|buf|
        llvm::LLVMBuildLandingPad(B(cx), Ty, PersFn, NumClauses, buf)
    });
}

fn SetCleanup(_cx: @block_ctxt, LandingPad: ValueRef) {
    llvm::LLVMSetCleanup(LandingPad, lib::llvm::True);
}

fn Resume(cx: @block_ctxt, Exn: ValueRef) -> ValueRef {
    assert (!cx.terminated);
    cx.terminated = true;
    ret llvm::LLVMBuildResume(B(cx), Exn);
}

//
// Local Variables:
// mode: rust
// fill-column: 78;
// indent-tabs-mode: nil
// c-basic-offset: 4
// buffer-file-coding-system: utf-8-unix
// compile-command: "make -k -C $RBUILD 2>&1 | sed -e 's/\\/x\\//x:\\//g'";
// End:
//
