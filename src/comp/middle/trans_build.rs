import core::{vec, str};
import core::ctypes::c_uint;
import str::sbuf;
import lib::llvm::llvm;
import syntax::codemap;
import codemap::span;
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
    unsafe {
        llvm::LLVMBuildAggregateRet(B(cx), vec::to_ptr(RetVals),
                                    vec::len(RetVals) as c_uint);
    }
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
    ret llvm::LLVMBuildSwitch(B(cx), V, Else, NumCases as c_uint);
}

fn AddCase(S: ValueRef, OnVal: ValueRef, Dest: BasicBlockRef) {
    if llvm::LLVMIsUndef(S) == lib::llvm::True { ret; }
    llvm::LLVMAddCase(S, OnVal, Dest);
}

fn IndirectBr(cx: @block_ctxt, Addr: ValueRef, NumDests: uint) {
    if cx.unreachable { ret; }
    assert (!cx.terminated);
    cx.terminated = true;
    llvm::LLVMBuildIndirectBr(B(cx), Addr, NumDests as c_uint);
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
        llvm::LLVMBuildInvoke(B(cx), Fn, vec::to_ptr(Args),
                              vec::len(Args) as c_uint, Then, Catch,
                              noname());
    }
}

fn FastInvoke(cx: @block_ctxt, Fn: ValueRef, Args: [ValueRef],
              Then: BasicBlockRef, Catch: BasicBlockRef) {
    if cx.unreachable { ret; }
    assert (!cx.terminated);
    cx.terminated = true;
    unsafe {
        let v = llvm::LLVMBuildInvoke(B(cx), Fn, vec::to_ptr(Args),
                                      vec::len(Args) as c_uint,
                                      Then, Catch, noname());
        llvm::LLVMSetInstructionCallConv(
            v, lib::llvm::LLVMFastCallConv as c_uint);
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
    ret llvm::LLVMBuildAdd(B(cx), LHS, RHS, noname());
}

fn NSWAdd(cx: @block_ctxt, LHS: ValueRef, RHS: ValueRef) -> ValueRef {
    if cx.unreachable { ret _Undef(LHS); }
    ret llvm::LLVMBuildNSWAdd(B(cx), LHS, RHS, noname());
}

fn NUWAdd(cx: @block_ctxt, LHS: ValueRef, RHS: ValueRef) -> ValueRef {
    if cx.unreachable { ret _Undef(LHS); }
    ret llvm::LLVMBuildNUWAdd(B(cx), LHS, RHS, noname());
}

fn FAdd(cx: @block_ctxt, LHS: ValueRef, RHS: ValueRef) -> ValueRef {
    if cx.unreachable { ret _Undef(LHS); }
    ret llvm::LLVMBuildFAdd(B(cx), LHS, RHS, noname());
}

fn Sub(cx: @block_ctxt, LHS: ValueRef, RHS: ValueRef) -> ValueRef {
    if cx.unreachable { ret _Undef(LHS); }
    ret llvm::LLVMBuildSub(B(cx), LHS, RHS, noname());
}

fn NSWSub(cx: @block_ctxt, LHS: ValueRef, RHS: ValueRef) -> ValueRef {
    if cx.unreachable { ret _Undef(LHS); }
    ret llvm::LLVMBuildNSWSub(B(cx), LHS, RHS, noname());
}

fn NUWSub(cx: @block_ctxt, LHS: ValueRef, RHS: ValueRef) -> ValueRef {
    if cx.unreachable { ret _Undef(LHS); }
    ret llvm::LLVMBuildNUWSub(B(cx), LHS, RHS, noname());
}

fn FSub(cx: @block_ctxt, LHS: ValueRef, RHS: ValueRef) -> ValueRef {
    if cx.unreachable { ret _Undef(LHS); }
    ret llvm::LLVMBuildFSub(B(cx), LHS, RHS, noname());
}

fn Mul(cx: @block_ctxt, LHS: ValueRef, RHS: ValueRef) -> ValueRef {
    if cx.unreachable { ret _Undef(LHS); }
    ret llvm::LLVMBuildMul(B(cx), LHS, RHS, noname());
}

fn NSWMul(cx: @block_ctxt, LHS: ValueRef, RHS: ValueRef) -> ValueRef {
    if cx.unreachable { ret _Undef(LHS); }
    ret llvm::LLVMBuildNSWMul(B(cx), LHS, RHS, noname());
}

fn NUWMul(cx: @block_ctxt, LHS: ValueRef, RHS: ValueRef) -> ValueRef {
    if cx.unreachable { ret _Undef(LHS); }
    ret llvm::LLVMBuildNUWMul(B(cx), LHS, RHS, noname());
}

fn FMul(cx: @block_ctxt, LHS: ValueRef, RHS: ValueRef) -> ValueRef {
    if cx.unreachable { ret _Undef(LHS); }
    ret llvm::LLVMBuildFMul(B(cx), LHS, RHS, noname());
}

fn UDiv(cx: @block_ctxt, LHS: ValueRef, RHS: ValueRef) -> ValueRef {
    if cx.unreachable { ret _Undef(LHS); }
    ret llvm::LLVMBuildUDiv(B(cx), LHS, RHS, noname());
}

fn SDiv(cx: @block_ctxt, LHS: ValueRef, RHS: ValueRef) -> ValueRef {
    if cx.unreachable { ret _Undef(LHS); }
    ret llvm::LLVMBuildSDiv(B(cx), LHS, RHS, noname());
}

fn ExactSDiv(cx: @block_ctxt, LHS: ValueRef, RHS: ValueRef) -> ValueRef {
    if cx.unreachable { ret _Undef(LHS); }
    ret llvm::LLVMBuildExactSDiv(B(cx), LHS, RHS, noname());
}

fn FDiv(cx: @block_ctxt, LHS: ValueRef, RHS: ValueRef) -> ValueRef {
    if cx.unreachable { ret _Undef(LHS); }
    ret llvm::LLVMBuildFDiv(B(cx), LHS, RHS, noname());
}

fn URem(cx: @block_ctxt, LHS: ValueRef, RHS: ValueRef) -> ValueRef {
    if cx.unreachable { ret _Undef(LHS); }
    ret llvm::LLVMBuildURem(B(cx), LHS, RHS, noname());
}

fn SRem(cx: @block_ctxt, LHS: ValueRef, RHS: ValueRef) -> ValueRef {
    if cx.unreachable { ret _Undef(LHS); }
    ret llvm::LLVMBuildSRem(B(cx), LHS, RHS, noname());
}

fn FRem(cx: @block_ctxt, LHS: ValueRef, RHS: ValueRef) -> ValueRef {
    if cx.unreachable { ret _Undef(LHS); }
    ret llvm::LLVMBuildFRem(B(cx), LHS, RHS, noname());
}

fn Shl(cx: @block_ctxt, LHS: ValueRef, RHS: ValueRef) -> ValueRef {
    if cx.unreachable { ret _Undef(LHS); }
    ret llvm::LLVMBuildShl(B(cx), LHS, RHS, noname());
}

fn LShr(cx: @block_ctxt, LHS: ValueRef, RHS: ValueRef) -> ValueRef {
    if cx.unreachable { ret _Undef(LHS); }
    ret llvm::LLVMBuildLShr(B(cx), LHS, RHS, noname());
}

fn AShr(cx: @block_ctxt, LHS: ValueRef, RHS: ValueRef) -> ValueRef {
    if cx.unreachable { ret _Undef(LHS); }
    ret llvm::LLVMBuildAShr(B(cx), LHS, RHS, noname());
}

fn And(cx: @block_ctxt, LHS: ValueRef, RHS: ValueRef) -> ValueRef {
    if cx.unreachable { ret _Undef(LHS); }
    ret llvm::LLVMBuildAnd(B(cx), LHS, RHS, noname());
}

fn Or(cx: @block_ctxt, LHS: ValueRef, RHS: ValueRef) -> ValueRef {
    if cx.unreachable { ret _Undef(LHS); }
    ret llvm::LLVMBuildOr(B(cx), LHS, RHS, noname());
}

fn Xor(cx: @block_ctxt, LHS: ValueRef, RHS: ValueRef) -> ValueRef {
    if cx.unreachable { ret _Undef(LHS); }
    ret llvm::LLVMBuildXor(B(cx), LHS, RHS, noname());
}

fn BinOp(cx: @block_ctxt, Op: Opcode, LHS: ValueRef, RHS: ValueRef) ->
   ValueRef {
    if cx.unreachable { ret _Undef(LHS); }
    ret llvm::LLVMBuildBinOp(B(cx), Op, LHS, RHS, noname());
}

fn Neg(cx: @block_ctxt, V: ValueRef) -> ValueRef {
    if cx.unreachable { ret _Undef(V); }
    ret llvm::LLVMBuildNeg(B(cx), V, noname());
}

fn NSWNeg(cx: @block_ctxt, V: ValueRef) -> ValueRef {
    if cx.unreachable { ret _Undef(V); }
    ret llvm::LLVMBuildNSWNeg(B(cx), V, noname());
}

fn NUWNeg(cx: @block_ctxt, V: ValueRef) -> ValueRef {
    if cx.unreachable { ret _Undef(V); }
    ret llvm::LLVMBuildNUWNeg(B(cx), V, noname());
}
fn FNeg(cx: @block_ctxt, V: ValueRef) -> ValueRef {
    if cx.unreachable { ret _Undef(V); }
    ret llvm::LLVMBuildFNeg(B(cx), V, noname());
}

fn Not(cx: @block_ctxt, V: ValueRef) -> ValueRef {
    if cx.unreachable { ret _Undef(V); }
    ret llvm::LLVMBuildNot(B(cx), V, noname());
}

/* Memory */
fn Malloc(cx: @block_ctxt, Ty: TypeRef) -> ValueRef {
    if cx.unreachable { ret llvm::LLVMGetUndef(T_ptr(T_i8())); }
    ret llvm::LLVMBuildMalloc(B(cx), Ty, noname());
}

fn ArrayMalloc(cx: @block_ctxt, Ty: TypeRef, Val: ValueRef) -> ValueRef {
    if cx.unreachable { ret llvm::LLVMGetUndef(T_ptr(T_i8())); }
    ret llvm::LLVMBuildArrayMalloc(B(cx), Ty, Val, noname());
}

fn Alloca(cx: @block_ctxt, Ty: TypeRef) -> ValueRef {
    if cx.unreachable { ret llvm::LLVMGetUndef(T_ptr(Ty)); }
    ret llvm::LLVMBuildAlloca(B(cx), Ty, noname());
}

fn ArrayAlloca(cx: @block_ctxt, Ty: TypeRef, Val: ValueRef) -> ValueRef {
    if cx.unreachable { ret llvm::LLVMGetUndef(T_ptr(Ty)); }
    ret llvm::LLVMBuildArrayAlloca(B(cx), Ty, Val, noname());
}

fn Free(cx: @block_ctxt, PointerVal: ValueRef) {
    if cx.unreachable { ret; }
    llvm::LLVMBuildFree(B(cx), PointerVal);
}

fn Load(cx: @block_ctxt, PointerVal: ValueRef) -> ValueRef {
    let ccx = cx.fcx.lcx.ccx;
    if cx.unreachable {
        let ty = val_ty(PointerVal);
        let eltty = if llvm::LLVMGetTypeKind(ty) == 11i32 {
            llvm::LLVMGetElementType(ty) } else { ccx.int_type };
        ret llvm::LLVMGetUndef(eltty);
    }
    ret llvm::LLVMBuildLoad(B(cx), PointerVal, noname());
}

fn Store(cx: @block_ctxt, Val: ValueRef, Ptr: ValueRef) {
    if cx.unreachable { ret; }
    llvm::LLVMBuildStore(B(cx), Val, Ptr);
}

fn GEP(cx: @block_ctxt, Pointer: ValueRef, Indices: [ValueRef]) -> ValueRef {
    if cx.unreachable { ret llvm::LLVMGetUndef(T_ptr(T_nil())); }
    unsafe {
        ret llvm::LLVMBuildGEP(B(cx), Pointer, vec::to_ptr(Indices),
                               vec::len(Indices) as c_uint, noname());
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
        ret llvm::LLVMBuildInBoundsGEP(B(cx), Pointer,
                                       vec::to_ptr(Indices),
                                       vec::len(Indices) as c_uint,
                                       noname());
    }
}

fn StructGEP(cx: @block_ctxt, Pointer: ValueRef, Idx: uint) -> ValueRef {
    if cx.unreachable { ret llvm::LLVMGetUndef(T_ptr(T_nil())); }
    ret llvm::LLVMBuildStructGEP(B(cx), Pointer, Idx as c_uint, noname());
}

fn GlobalString(cx: @block_ctxt, _Str: sbuf) -> ValueRef {
    if cx.unreachable { ret llvm::LLVMGetUndef(T_ptr(T_i8())); }
    ret llvm::LLVMBuildGlobalString(B(cx), _Str, noname());
}

fn GlobalStringPtr(cx: @block_ctxt, _Str: sbuf) -> ValueRef {
    if cx.unreachable { ret llvm::LLVMGetUndef(T_ptr(T_i8())); }
    ret llvm::LLVMBuildGlobalStringPtr(B(cx), _Str, noname());
}

/* Casts */
fn Trunc(cx: @block_ctxt, Val: ValueRef, DestTy: TypeRef) -> ValueRef {
    if cx.unreachable { ret llvm::LLVMGetUndef(DestTy); }
    ret llvm::LLVMBuildTrunc(B(cx), Val, DestTy, noname());
}

fn ZExt(cx: @block_ctxt, Val: ValueRef, DestTy: TypeRef) -> ValueRef {
    if cx.unreachable { ret llvm::LLVMGetUndef(DestTy); }
    ret llvm::LLVMBuildZExt(B(cx), Val, DestTy, noname());
}

fn SExt(cx: @block_ctxt, Val: ValueRef, DestTy: TypeRef) -> ValueRef {
    if cx.unreachable { ret llvm::LLVMGetUndef(DestTy); }
    ret llvm::LLVMBuildSExt(B(cx), Val, DestTy, noname());
}

fn FPToUI(cx: @block_ctxt, Val: ValueRef, DestTy: TypeRef) -> ValueRef {
    if cx.unreachable { ret llvm::LLVMGetUndef(DestTy); }
    ret llvm::LLVMBuildFPToUI(B(cx), Val, DestTy, noname());
}

fn FPToSI(cx: @block_ctxt, Val: ValueRef, DestTy: TypeRef) -> ValueRef {
    if cx.unreachable { ret llvm::LLVMGetUndef(DestTy); }
    ret llvm::LLVMBuildFPToSI(B(cx), Val, DestTy, noname());
}

fn UIToFP(cx: @block_ctxt, Val: ValueRef, DestTy: TypeRef) -> ValueRef {
    if cx.unreachable { ret llvm::LLVMGetUndef(DestTy); }
    ret llvm::LLVMBuildUIToFP(B(cx), Val, DestTy, noname());
}

fn SIToFP(cx: @block_ctxt, Val: ValueRef, DestTy: TypeRef) -> ValueRef {
    if cx.unreachable { ret llvm::LLVMGetUndef(DestTy); }
    ret llvm::LLVMBuildSIToFP(B(cx), Val, DestTy, noname());
}

fn FPTrunc(cx: @block_ctxt, Val: ValueRef, DestTy: TypeRef) -> ValueRef {
    if cx.unreachable { ret llvm::LLVMGetUndef(DestTy); }
    ret llvm::LLVMBuildFPTrunc(B(cx), Val, DestTy, noname());
}

fn FPExt(cx: @block_ctxt, Val: ValueRef, DestTy: TypeRef) -> ValueRef {
    if cx.unreachable { ret llvm::LLVMGetUndef(DestTy); }
    ret llvm::LLVMBuildFPExt(B(cx), Val, DestTy, noname());
}

fn PtrToInt(cx: @block_ctxt, Val: ValueRef, DestTy: TypeRef) -> ValueRef {
    if cx.unreachable { ret llvm::LLVMGetUndef(DestTy); }
    ret llvm::LLVMBuildPtrToInt(B(cx), Val, DestTy, noname());
}

fn IntToPtr(cx: @block_ctxt, Val: ValueRef, DestTy: TypeRef) -> ValueRef {
    if cx.unreachable { ret llvm::LLVMGetUndef(DestTy); }
    ret llvm::LLVMBuildIntToPtr(B(cx), Val, DestTy, noname());
}

fn BitCast(cx: @block_ctxt, Val: ValueRef, DestTy: TypeRef) -> ValueRef {
    if cx.unreachable { ret llvm::LLVMGetUndef(DestTy); }
    ret llvm::LLVMBuildBitCast(B(cx), Val, DestTy, noname());
}

fn ZExtOrBitCast(cx: @block_ctxt, Val: ValueRef, DestTy: TypeRef) ->
   ValueRef {
    if cx.unreachable { ret llvm::LLVMGetUndef(DestTy); }
    ret llvm::LLVMBuildZExtOrBitCast(B(cx), Val, DestTy, noname());
}

fn SExtOrBitCast(cx: @block_ctxt, Val: ValueRef, DestTy: TypeRef) ->
   ValueRef {
    if cx.unreachable { ret llvm::LLVMGetUndef(DestTy); }
    ret llvm::LLVMBuildSExtOrBitCast(B(cx), Val, DestTy, noname());
}

fn TruncOrBitCast(cx: @block_ctxt, Val: ValueRef, DestTy: TypeRef) ->
   ValueRef {
    if cx.unreachable { ret llvm::LLVMGetUndef(DestTy); }
    ret llvm::LLVMBuildTruncOrBitCast(B(cx), Val, DestTy, noname());
}

fn Cast(cx: @block_ctxt, Op: Opcode, Val: ValueRef, DestTy: TypeRef,
        _Name: sbuf) -> ValueRef {
    if cx.unreachable { ret llvm::LLVMGetUndef(DestTy); }
    ret llvm::LLVMBuildCast(B(cx), Op, Val, DestTy, noname());
}

fn PointerCast(cx: @block_ctxt, Val: ValueRef, DestTy: TypeRef) -> ValueRef {
    if cx.unreachable { ret llvm::LLVMGetUndef(DestTy); }
    ret llvm::LLVMBuildPointerCast(B(cx), Val, DestTy, noname());
}

fn IntCast(cx: @block_ctxt, Val: ValueRef, DestTy: TypeRef) -> ValueRef {
    if cx.unreachable { ret llvm::LLVMGetUndef(DestTy); }
    ret llvm::LLVMBuildIntCast(B(cx), Val, DestTy, noname());
}

fn FPCast(cx: @block_ctxt, Val: ValueRef, DestTy: TypeRef) -> ValueRef {
    if cx.unreachable { ret llvm::LLVMGetUndef(DestTy); }
    ret llvm::LLVMBuildFPCast(B(cx), Val, DestTy, noname());
}


/* Comparisons */
fn ICmp(cx: @block_ctxt, Op: uint, LHS: ValueRef, RHS: ValueRef) -> ValueRef {
    if cx.unreachable { ret llvm::LLVMGetUndef(T_i1()); }
    ret llvm::LLVMBuildICmp(B(cx), Op as c_uint, LHS, RHS, noname());
}

fn FCmp(cx: @block_ctxt, Op: uint, LHS: ValueRef, RHS: ValueRef) -> ValueRef {
    if cx.unreachable { ret llvm::LLVMGetUndef(T_i1()); }
    ret llvm::LLVMBuildFCmp(B(cx), Op as c_uint, LHS, RHS, noname());
}

/* Miscellaneous instructions */
fn EmptyPhi(cx: @block_ctxt, Ty: TypeRef) -> ValueRef {
    if cx.unreachable { ret llvm::LLVMGetUndef(Ty); }
    ret llvm::LLVMBuildPhi(B(cx), Ty, noname());
}

fn Phi(cx: @block_ctxt, Ty: TypeRef, vals: [ValueRef], bbs: [BasicBlockRef])
   -> ValueRef {
    if cx.unreachable { ret llvm::LLVMGetUndef(Ty); }
    assert (vec::len::<ValueRef>(vals) == vec::len::<BasicBlockRef>(bbs));
    let phi = EmptyPhi(cx, Ty);
    unsafe {
        llvm::LLVMAddIncoming(phi, vec::to_ptr(vals), vec::to_ptr(bbs),
                              vec::len(vals) as c_uint);
        ret phi;
    }
}

fn AddIncomingToPhi(phi: ValueRef, val: ValueRef, bb: BasicBlockRef) {
    if llvm::LLVMIsUndef(phi) == lib::llvm::True { ret; }
    unsafe {
        let valptr = unsafe::reinterpret_cast(ptr::addr_of(val));
        let bbptr = unsafe::reinterpret_cast(ptr::addr_of(bb));
        llvm::LLVMAddIncoming(phi, valptr, bbptr, 1u32);
    }
}

fn _UndefReturn(cx: @block_ctxt, Fn: ValueRef) -> ValueRef {
    let ccx = cx.fcx.lcx.ccx;
    let ty = val_ty(Fn);
    let retty = if llvm::LLVMGetTypeKind(ty) == 8i32 {
        llvm::LLVMGetReturnType(ty) } else { ccx.int_type };
    ret llvm::LLVMGetUndef(retty);
}

fn add_span_comment(bcx: @block_ctxt, sp: span, text: str) {
    let ccx = bcx_ccx(bcx);
    if (!ccx.sess.opts.no_asm_comments) {
        let s = text + " (" + codemap::span_to_str(sp, ccx.sess.codemap)
            + ")";
        log(debug, s);
        add_comment(bcx, s);
    }
}

fn add_comment(bcx: @block_ctxt, text: str) {
    let ccx = bcx_ccx(bcx);
    if (!ccx.sess.opts.no_asm_comments) {
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
        ret llvm::LLVMBuildCall(B(cx), Fn, vec::to_ptr(Args),
                                vec::len(Args) as c_uint, noname());
    }
}

fn FastCall(cx: @block_ctxt, Fn: ValueRef, Args: [ValueRef]) -> ValueRef {
    if cx.unreachable { ret _UndefReturn(cx, Fn); }
    unsafe {
        let v = llvm::LLVMBuildCall(B(cx), Fn, vec::to_ptr(Args),
                                    vec::len(Args) as c_uint, noname());
        llvm::LLVMSetInstructionCallConv(
            v, lib::llvm::LLVMFastCallConv as c_uint);
        ret v;
    }
}

fn CallWithConv(cx: @block_ctxt, Fn: ValueRef, Args: [ValueRef],
                Conv: c_uint) -> ValueRef {
    if cx.unreachable { ret _UndefReturn(cx, Fn); }
    unsafe {
        let v = llvm::LLVMBuildCall(B(cx), Fn, vec::to_ptr(Args),
                                    vec::len(Args) as c_uint, noname());
        llvm::LLVMSetInstructionCallConv(v, Conv);
        ret v;
    }
}

fn Select(cx: @block_ctxt, If: ValueRef, Then: ValueRef, Else: ValueRef) ->
   ValueRef {
    if cx.unreachable { ret _Undef(Then); }
    ret llvm::LLVMBuildSelect(B(cx), If, Then, Else, noname());
}

fn VAArg(cx: @block_ctxt, list: ValueRef, Ty: TypeRef) -> ValueRef {
    if cx.unreachable { ret llvm::LLVMGetUndef(Ty); }
    ret llvm::LLVMBuildVAArg(B(cx), list, Ty, noname());
}

fn ExtractElement(cx: @block_ctxt, VecVal: ValueRef, Index: ValueRef) ->
   ValueRef {
    if cx.unreachable { ret llvm::LLVMGetUndef(T_nil()); }
    ret llvm::LLVMBuildExtractElement(B(cx), VecVal, Index, noname());
}

fn InsertElement(cx: @block_ctxt, VecVal: ValueRef, EltVal: ValueRef,
                 Index: ValueRef) {
    if cx.unreachable { ret; }
    llvm::LLVMBuildInsertElement(B(cx), VecVal, EltVal, Index, noname());
}

fn ShuffleVector(cx: @block_ctxt, V1: ValueRef, V2: ValueRef,
                 Mask: ValueRef) {
    if cx.unreachable { ret; }
    llvm::LLVMBuildShuffleVector(B(cx), V1, V2, Mask, noname());
}

fn ExtractValue(cx: @block_ctxt, AggVal: ValueRef, Index: uint) -> ValueRef {
    if cx.unreachable { ret llvm::LLVMGetUndef(T_nil()); }
    ret llvm::LLVMBuildExtractValue(B(cx), AggVal, Index as c_uint, noname());
}

fn InsertValue(cx: @block_ctxt, AggVal: ValueRef, EltVal: ValueRef,
               Index: uint) {
    if cx.unreachable { ret; }
    llvm::LLVMBuildInsertValue(B(cx), AggVal, EltVal, Index as c_uint,
                               noname());
}

fn IsNull(cx: @block_ctxt, Val: ValueRef) -> ValueRef {
    if cx.unreachable { ret llvm::LLVMGetUndef(T_i1()); }
    ret llvm::LLVMBuildIsNull(B(cx), Val, noname());
}

fn IsNotNull(cx: @block_ctxt, Val: ValueRef) -> ValueRef {
    if cx.unreachable { ret llvm::LLVMGetUndef(T_i1()); }
    ret llvm::LLVMBuildIsNotNull(B(cx), Val, noname());
}

fn PtrDiff(cx: @block_ctxt, LHS: ValueRef, RHS: ValueRef) -> ValueRef {
    let ccx = cx.fcx.lcx.ccx;
    if cx.unreachable { ret llvm::LLVMGetUndef(ccx.int_type); }
    ret llvm::LLVMBuildPtrDiff(B(cx), LHS, RHS, noname());
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
        llvm::LLVMBuildCall(b, T, vec::to_ptr(Args),
                            vec::len(Args) as c_uint, noname());
    }
}

fn LandingPad(cx: @block_ctxt, Ty: TypeRef, PersFn: ValueRef,
              NumClauses: uint) -> ValueRef {
    assert !cx.terminated && !cx.unreachable;
    ret llvm::LLVMBuildLandingPad(B(cx), Ty, PersFn,
                                  NumClauses as c_uint, noname());
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
// End:
//
