#![allow(non_camel_case_types)]

use super::ffi::*;
use libc::{c_char, c_uint, size_t};

use rustc_ast::expand::autodiff_attrs::DiffActivity;
use tracing::trace;

extern "C" {
    // Enzyme
    pub fn LLVMRustAddFncParamAttr<'a>(F: &'a Value, index: c_uint, Attr: &'a Attribute);

    pub fn LLVMRustAddRetFncAttr(F: &Value, attr: &Attribute);
    pub fn LLVMRustHasMetadata(I: &Value, KindID: c_uint) -> bool;
    pub fn LLVMRustEraseInstBefore(BB: &BasicBlock, I: &Value);
    pub fn LLVMRustGetLastInstruction<'a>(BB: &BasicBlock) -> Option<&'a Value>;
    pub fn LLVMRustDIGetInstMetadata(I: &Value) -> &Metadata;
    pub fn LLVMRustEraseInstFromParent(V: &Value);
    pub fn LLVMRustGetTerminator<'a>(B: &BasicBlock) -> &'a Value;
    pub fn LLVMGetReturnType(T: &Type) -> &Type;
    pub fn LLVMRustIsStructType(T: &Type) -> bool;
    pub fn LLVMDumpModule(M: &Module);
    pub fn LLVMCountStructElementTypes(T: &Type) -> c_uint;
    pub fn LLVMVerifyFunction(V: &Value, action: LLVMVerifierFailureAction) -> bool;
    pub fn LLVMGetParams(Fnc: &Value, parms: *mut &Value);
    pub fn LLVMBuildCall2<'a>(
        arg1: &Builder<'a>,
        ty: &Type,
        func: &Value,
        args: *mut &Value,
        num_args: size_t,
        name: *const c_char,
    ) -> &'a Value;
    pub fn LLVMGetFirstFunction(M: &Module) -> Option<&Value>;
    pub fn LLVMGetNextFunction(V: &Value) -> Option<&Value>;
    pub fn LLVMGetNamedFunction(M: &Module, Name: *const c_char) -> Option<&Value>;
    pub fn LLVMGlobalGetValueType(val: &Value) -> &Type;
    pub fn LLVMRustGetFunctionType(fnc: &Value) -> &Type;

    pub fn LLVMRemoveStringAttributeAtIndex(F: &Value, Idx: c_uint, K: *const c_char, KLen: c_uint);
    pub fn LLVMGetStringAttributeAtIndex(
        F: &Value,
        Idx: c_uint,
        K: *const c_char,
        KLen: c_uint,
    ) -> &Attribute;
    pub fn LLVMIsEnumAttribute(A: &Attribute) -> bool;
    pub fn LLVMIsStringAttribute(A: &Attribute) -> bool;
    pub fn LLVMRustAddEnumAttributeAtIndex(
        C: &Context,
        V: &Value,
        index: c_uint,
        attr: AttributeKind,
    );
    pub fn LLVMRustRemoveEnumAttributeAtIndex(V: &Value, index: c_uint, attr: AttributeKind);
    pub fn LLVMRustGetEnumAttributeAtIndex(
        V: &Value,
        index: c_uint,
        attr: AttributeKind,
    ) -> &Attribute;

    pub fn LLVMRustAddParamAttr<'a>(Instr: &'a Value, index: c_uint, Attr: &'a Attribute);

}

#[repr(C)]
pub enum LLVMVerifierFailureAction {
    LLVMAbortProcessAction,
    LLVMPrintMessageAction,
    LLVMReturnStatusAction,
}

#[allow(unused_unsafe)]
pub(crate) unsafe fn enzyme_rust_forward_diff(
    logic_ref: EnzymeLogicRef,
    type_analysis: EnzymeTypeAnalysisRef,
    fnc: &Value,
    input_diffactivity: Vec<DiffActivity>,
    ret_diffactivity: DiffActivity,
    void_ret: bool,
) -> (&Value, Vec<usize>) {
    let ret_activity = cdiffe_from(ret_diffactivity);
    assert!(ret_activity != CDIFFE_TYPE::DFT_OUT_DIFF);
    let mut input_activity: Vec<CDIFFE_TYPE> = vec![];
    for input in input_diffactivity {
        let act = cdiffe_from(input);
        assert!(
            act == CDIFFE_TYPE::DFT_CONSTANT
                || act == CDIFFE_TYPE::DFT_DUP_ARG
                || act == CDIFFE_TYPE::DFT_DUP_NONEED
        );
        input_activity.push(act);
    }

    // if we have void ret, this must be false;
    let ret_primary_ret = if void_ret {
        false
    } else {
        match ret_activity {
            CDIFFE_TYPE::DFT_CONSTANT => true,
            CDIFFE_TYPE::DFT_DUP_ARG => true,
            CDIFFE_TYPE::DFT_DUP_NONEED => false,
            _ => panic!("Implementation error in enzyme_rust_forward_diff."),
        }
    };
    trace!("ret_primary_ret: {}", &ret_primary_ret);

    // We don't support volatile / extern / (global?) values.
    // Just because I didn't had time to test them, and it seems less urgent.
    let args_uncacheable = vec![0; input_activity.len()];
    let num_fnc_args = unsafe { LLVMCountParams(fnc) };
    trace!("num_fnc_args: {}", num_fnc_args);
    trace!("input_activity.len(): {}", input_activity.len());
    assert!(num_fnc_args == input_activity.len() as u32);

    let kv_tmp = IntList { data: std::ptr::null_mut(), size: 0 };

    let mut known_values = vec![kv_tmp; input_activity.len()];

    let tree_tmp = TypeTree::new();
    let mut args_tree = vec![tree_tmp.inner; input_activity.len()];

    let ret_tt = TypeTree::new();
    let dummy_type = CFnTypeInfo {
        Arguments: args_tree.as_mut_ptr(),
        Return: ret_tt.inner,
        KnownValues: known_values.as_mut_ptr(),
    };

    trace!("ret_activity: {}", &ret_activity);
    for i in &input_activity {
        trace!("input_activity i: {}", &i);
    }
    trace!("before calling Enzyme");
    let res = unsafe {
        EnzymeCreateForwardDiff(
            logic_ref, // Logic
            std::ptr::null(),
            std::ptr::null(),
            fnc,
            ret_activity, // LLVM function, return type
            input_activity.as_ptr(),
            input_activity.len(), // constant arguments
            type_analysis,        // type analysis struct
            ret_primary_ret as u8,
            CDerivativeMode::DEM_ForwardMode, // return value, dret_used, top_level which was 1
            1,                                // free memory
            1,                                // vector mode width
            Option::None,
            dummy_type, // additional_arg, type info (return + args)
            args_uncacheable.as_ptr(),
            args_uncacheable.len(), // uncacheable arguments
            std::ptr::null_mut(),   // write augmented function to this
        )
    };
    trace!("after calling Enzyme");
    (res, vec![])
}

#[allow(unused_unsafe)]
pub(crate) unsafe fn enzyme_rust_reverse_diff(
    logic_ref: EnzymeLogicRef,
    type_analysis: EnzymeTypeAnalysisRef,
    fnc: &Value,
    rust_input_activity: Vec<DiffActivity>,
    ret_activity: DiffActivity,
) -> (&Value, Vec<usize>) {
    let (primary_ret, ret_activity) = match ret_activity {
        DiffActivity::Const => (true, CDIFFE_TYPE::DFT_CONSTANT),
        DiffActivity::Active => (true, CDIFFE_TYPE::DFT_OUT_DIFF),
        DiffActivity::ActiveOnly => (false, CDIFFE_TYPE::DFT_OUT_DIFF),
        DiffActivity::None => (false, CDIFFE_TYPE::DFT_CONSTANT),
        _ => panic!("Invalid return activity"),
    };
    // This only is needed for split-mode AD, which we don't support.
    // See Julia:
    // https://github.com/EnzymeAD/Enzyme.jl/blob/a511e4e6979d6161699f5c9919d49801c0764a09/src/compiler.jl#L3132
    // https://github.com/EnzymeAD/Enzyme.jl/blob/a511e4e6979d6161699f5c9919d49801c0764a09/src/compiler.jl#L3092
    let diff_ret = false;

    let mut primal_sizes = vec![];
    let mut input_activity: Vec<CDIFFE_TYPE> = vec![];
    for (i, &x) in rust_input_activity.iter().enumerate() {
        if is_size(x) {
            primal_sizes.push(i);
            input_activity.push(CDIFFE_TYPE::DFT_CONSTANT);
            continue;
        }
        input_activity.push(cdiffe_from(x));
    }

    // We don't support volatile / extern / (global?) values.
    // Just because I didn't had time to test them, and it seems less urgent.
    let args_uncacheable = vec![0; input_activity.len()];
    let num_fnc_args = unsafe { LLVMCountParams(fnc) };
    println!("num_fnc_args: {}", num_fnc_args);
    println!("input_activity.len(): {}", input_activity.len());
    assert!(num_fnc_args == input_activity.len() as u32);
    let kv_tmp = IntList { data: std::ptr::null_mut(), size: 0 };

    let mut known_values = vec![kv_tmp; input_activity.len()];

    let tree_tmp = TypeTree::new();
    let mut args_tree = vec![tree_tmp.inner; input_activity.len()];
    let ret_tt = TypeTree::new();
    let dummy_type = CFnTypeInfo {
        Arguments: args_tree.as_mut_ptr(),
        Return: ret_tt.inner,
        KnownValues: known_values.as_mut_ptr(),
    };

    trace!("primary_ret: {}", &primary_ret);
    trace!("ret_activity: {}", &ret_activity);
    for i in &input_activity {
        trace!("input_activity i: {}", &i);
    }
    trace!("before calling Enzyme");
    let res = unsafe {
        EnzymeCreatePrimalAndGradient(
            logic_ref, // Logic
            std::ptr::null(),
            std::ptr::null(),
            fnc,
            ret_activity, // LLVM function, return type
            input_activity.as_ptr(),
            input_activity.len(), // constant arguments
            type_analysis,        // type analysis struct
            primary_ret as u8,
            diff_ret as u8,                           //0
            CDerivativeMode::DEM_ReverseModeCombined, // return value, dret_used, top_level which was 1
            1,                                        // vector mode width
            1,                                        // free memory
            Option::None,
            0,          // do not force anonymous tape
            dummy_type, // additional_arg, type info (return + args)
            args_uncacheable.as_ptr(),
            args_uncacheable.len(), // uncacheable arguments
            std::ptr::null_mut(),   // write augmented function to this
            0,
        )
    };
    trace!("after calling Enzyme");
    (res, primal_sizes)
}

#[cfg(not(llvm_enzyme))]
pub use self::Fallback_AD::*;

#[cfg(not(llvm_enzyme))]
pub mod Fallback_AD {
    #![allow(unused_variables)]
    use super::*;

    pub fn EnzymeNewTypeTree() -> CTypeTreeRef {
        unimplemented!()
    }
    pub fn EnzymeFreeTypeTree(CTT: CTypeTreeRef) {
        unimplemented!()
    }
    pub fn EnzymeSetCLBool(arg1: *mut ::std::os::raw::c_void, arg2: u8) {
        unimplemented!()
    }
    pub fn EnzymeSetCLInteger(arg1: *mut ::std::os::raw::c_void, arg2: i64) {
        unimplemented!()
    }

    pub fn set_inline(val: bool) {
        unimplemented!()
    }
    pub fn set_runtime_activity_check(check: bool) {
        unimplemented!()
    }
    pub fn set_max_int_offset(offset: u64) {
        unimplemented!()
    }
    pub fn set_max_type_offset(offset: u64) {
        unimplemented!()
    }
    pub fn set_max_type_depth(depth: u64) {
        unimplemented!()
    }
    pub fn set_print_perf(print: bool) {
        unimplemented!()
    }
    pub fn set_print_activity(print: bool) {
        unimplemented!()
    }
    pub fn set_print_type(print: bool) {
        unimplemented!()
    }
    pub fn set_print(print: bool) {
        unimplemented!()
    }
    pub fn set_strict_aliasing(strict: bool) {
        unimplemented!()
    }
    pub fn set_loose_types(loose: bool) {
        unimplemented!()
    }

    pub fn EnzymeCreatePrimalAndGradient<'a>(
        arg1: EnzymeLogicRef,
        _builderCtx: *const u8, // &'a Builder<'_>,
        _callerCtx: *const u8,  // &'a Value,
        todiff: &'a Value,
        retType: CDIFFE_TYPE,
        constant_args: *const CDIFFE_TYPE,
        constant_args_size: size_t,
        TA: EnzymeTypeAnalysisRef,
        returnValue: u8,
        dretUsed: u8,
        mode: CDerivativeMode,
        width: ::std::os::raw::c_uint,
        freeMemory: u8,
        additionalArg: Option<&Type>,
        forceAnonymousTape: u8,
        typeInfo: CFnTypeInfo,
        _uncacheable_args: *const u8,
        uncacheable_args_size: size_t,
        augmented: EnzymeAugmentedReturnPtr,
        AtomicAdd: u8,
    ) -> &'a Value {
        unimplemented!()
    }
    pub fn EnzymeCreateForwardDiff<'a>(
        arg1: EnzymeLogicRef,
        _builderCtx: *const u8, // &'a Builder<'_>,
        _callerCtx: *const u8,  // &'a Value,
        todiff: &'a Value,
        retType: CDIFFE_TYPE,
        constant_args: *const CDIFFE_TYPE,
        constant_args_size: size_t,
        TA: EnzymeTypeAnalysisRef,
        returnValue: u8,
        mode: CDerivativeMode,
        freeMemory: u8,
        width: ::std::os::raw::c_uint,
        additionalArg: Option<&Type>,
        typeInfo: CFnTypeInfo,
        _uncacheable_args: *const u8,
        uncacheable_args_size: size_t,
        augmented: EnzymeAugmentedReturnPtr,
    ) -> &'a Value {
        unimplemented!()
    }
    pub type CustomRuleType = ::std::option::Option<
        unsafe extern "C" fn(
            direction: ::std::os::raw::c_int,
            ret: CTypeTreeRef,
            args: *mut CTypeTreeRef,
            known_values: *mut IntList,
            num_args: size_t,
            fnc: &Value,
            ta: *const ::std::os::raw::c_void,
        ) -> u8,
    >;
    extern "C" {
        pub fn CreateTypeAnalysis(
            Log: EnzymeLogicRef,
            customRuleNames: *mut *mut ::std::os::raw::c_char,
            customRules: *mut CustomRuleType,
            numRules: size_t,
        ) -> EnzymeTypeAnalysisRef;
    }
    //pub fn ClearTypeAnalysis(arg1: EnzymeTypeAnalysisRef) { unimplemented!() }
    pub fn FreeTypeAnalysis(arg1: EnzymeTypeAnalysisRef) {
        unimplemented!()
    }
    pub fn CreateEnzymeLogic(PostOpt: u8) -> EnzymeLogicRef {
        unimplemented!()
    }
    pub fn ClearEnzymeLogic(arg1: EnzymeLogicRef) {
        unimplemented!()
    }
    pub fn FreeEnzymeLogic(arg1: EnzymeLogicRef) {
        unimplemented!()
    }

    pub fn EnzymeNewTypeTreeCT(arg1: CConcreteType, ctx: &Context) -> CTypeTreeRef {
        unimplemented!()
    }
    pub fn EnzymeNewTypeTreeTR(arg1: CTypeTreeRef) -> CTypeTreeRef {
        unimplemented!()
    }
    pub fn EnzymeMergeTypeTree(arg1: CTypeTreeRef, arg2: CTypeTreeRef) -> bool {
        unimplemented!()
    }
    pub fn EnzymeTypeTreeOnlyEq(arg1: CTypeTreeRef, pos: i64) {
        unimplemented!()
    }
    pub fn EnzymeTypeTreeData0Eq(arg1: CTypeTreeRef) {
        unimplemented!()
    }
    pub fn EnzymeTypeTreeShiftIndiciesEq(
        arg1: CTypeTreeRef,
        data_layout: *const c_char,
        offset: i64,
        max_size: i64,
        add_offset: u64,
    ) {
        unimplemented!()
    }
    pub fn EnzymeTypeTreeToStringFree(arg1: *const c_char) {
        unimplemented!()
    }
    pub fn EnzymeTypeTreeToString(arg1: CTypeTreeRef) -> *const c_char {
        unimplemented!()
    }
}

// Enzyme specific, but doesn't require Enzyme to be build
pub use self::Shared_AD::*;
pub mod Shared_AD {
    // Depending on the AD backend (Enzyme or Fallback), some functions might or might not be
    // unsafe. So we just allways call them in an unsafe context.
    #![allow(unused_unsafe)]
    #![allow(unused_variables)]

    use core::fmt;
    use std::ffi::{CStr, CString};

    use libc::size_t;
    use rustc_ast::expand::autodiff_attrs::DiffActivity;

    use super::Context;
    #[cfg(llvm_enzyme)]
    use super::Enzyme_AD::*;
    #[cfg(not(llvm_enzyme))]
    use super::Fallback_AD::*;
    #[repr(u32)]
    #[derive(Debug, Copy, Clone, Hash, PartialEq, Eq)]
    pub enum CDIFFE_TYPE {
        DFT_OUT_DIFF = 0,
        DFT_DUP_ARG = 1,
        DFT_CONSTANT = 2,
        DFT_DUP_NONEED = 3,
    }

    impl fmt::Display for CDIFFE_TYPE {
        fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
            let value = match self {
                CDIFFE_TYPE::DFT_OUT_DIFF => "DFT_OUT_DIFF",
                CDIFFE_TYPE::DFT_DUP_ARG => "DFT_DUP_ARG",
                CDIFFE_TYPE::DFT_CONSTANT => "DFT_CONSTANT",
                CDIFFE_TYPE::DFT_DUP_NONEED => "DFT_DUP_NONEED",
            };
            write!(f, "{}", value)
        }
    }

    pub fn cdiffe_from(act: DiffActivity) -> CDIFFE_TYPE {
        return match act {
            DiffActivity::None => CDIFFE_TYPE::DFT_CONSTANT,
            DiffActivity::Const => CDIFFE_TYPE::DFT_CONSTANT,
            DiffActivity::Active => CDIFFE_TYPE::DFT_OUT_DIFF,
            DiffActivity::ActiveOnly => CDIFFE_TYPE::DFT_OUT_DIFF,
            DiffActivity::Dual => CDIFFE_TYPE::DFT_DUP_ARG,
            DiffActivity::DualOnly => CDIFFE_TYPE::DFT_DUP_NONEED,
            DiffActivity::Duplicated => CDIFFE_TYPE::DFT_DUP_ARG,
            DiffActivity::DuplicatedOnly => CDIFFE_TYPE::DFT_DUP_NONEED,
            DiffActivity::FakeActivitySize => panic!("Implementation error"),
        };
    }

    pub fn is_size(act: DiffActivity) -> bool {
        return act == DiffActivity::FakeActivitySize;
    }

    #[repr(u32)]
    #[derive(Debug, Copy, Clone, Hash, PartialEq, Eq)]
    pub enum CDerivativeMode {
        DEM_ForwardMode = 0,
        DEM_ReverseModePrimal = 1,
        DEM_ReverseModeGradient = 2,
        DEM_ReverseModeCombined = 3,
        DEM_ForwardModeSplit = 4,
    }
    #[repr(C)]
    #[derive(Debug, Copy, Clone)]
    pub struct EnzymeOpaqueTypeAnalysis {
        _unused: [u8; 0],
    }
    pub type EnzymeTypeAnalysisRef = *mut EnzymeOpaqueTypeAnalysis;
    #[repr(C)]
    #[derive(Debug, Copy, Clone)]
    pub struct EnzymeOpaqueLogic {
        _unused: [u8; 0],
    }
    pub type EnzymeLogicRef = *mut EnzymeOpaqueLogic;
    #[repr(C)]
    #[derive(Debug, Copy, Clone)]
    pub struct EnzymeOpaqueAugmentedReturn {
        _unused: [u8; 0],
    }
    pub type EnzymeAugmentedReturnPtr = *mut EnzymeOpaqueAugmentedReturn;
    #[repr(C)]
    #[derive(Debug, Copy, Clone)]
    pub struct IntList {
        pub data: *mut i64,
        pub size: size_t,
    }
    #[repr(u32)]
    #[derive(Debug, Copy, Clone, Hash, PartialEq, Eq)]
    pub enum CConcreteType {
        DT_Anything = 0,
        DT_Integer = 1,
        DT_Pointer = 2,
        DT_Half = 3,
        DT_Float = 4,
        DT_Double = 5,
        DT_Unknown = 6,
    }

    pub type CTypeTreeRef = *mut EnzymeTypeTree;
    #[repr(C)]
    #[derive(Debug, Copy, Clone)]
    pub struct EnzymeTypeTree {
        _unused: [u8; 0],
    }
    pub struct TypeTree {
        pub inner: CTypeTreeRef,
    }

    impl TypeTree {
        pub fn new() -> TypeTree {
            let inner = unsafe { EnzymeNewTypeTree() };
            TypeTree { inner }
        }

        #[must_use]
        pub fn from_type(t: CConcreteType, ctx: &Context) -> TypeTree {
            let inner = unsafe { EnzymeNewTypeTreeCT(t, ctx) };
            TypeTree { inner }
        }

        #[must_use]
        pub fn only(self, idx: isize) -> TypeTree {
            unsafe {
                EnzymeTypeTreeOnlyEq(self.inner, idx as i64);
            }
            self
        }

        #[must_use]
        pub fn data0(self) -> TypeTree {
            unsafe {
                EnzymeTypeTreeData0Eq(self.inner);
            }
            self
        }

        pub fn merge(self, other: Self) -> Self {
            unsafe {
                EnzymeMergeTypeTree(self.inner, other.inner);
            }
            drop(other);
            self
        }

        #[must_use]
        pub fn shift(
            self,
            layout: &str,
            offset: isize,
            max_size: isize,
            add_offset: usize,
        ) -> Self {
            let layout = CString::new(layout).unwrap();

            unsafe {
                EnzymeTypeTreeShiftIndiciesEq(
                    self.inner,
                    layout.as_ptr(),
                    offset as i64,
                    max_size as i64,
                    add_offset as u64,
                )
            }

            self
        }
    }

    impl Clone for TypeTree {
        fn clone(&self) -> Self {
            let inner = unsafe { EnzymeNewTypeTreeTR(self.inner) };
            TypeTree { inner }
        }
    }

    impl fmt::Display for TypeTree {
        fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
            let ptr = unsafe { EnzymeTypeTreeToString(self.inner) };
            let cstr = unsafe { CStr::from_ptr(ptr) };
            match cstr.to_str() {
                Ok(x) => write!(f, "{}", x)?,
                Err(err) => write!(f, "could not parse: {}", err)?,
            }

            // delete C string pointer
            unsafe { EnzymeTypeTreeToStringFree(ptr) }

            Ok(())
        }
    }

    impl fmt::Debug for TypeTree {
        fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
            <Self as fmt::Display>::fmt(self, f)
        }
    }

    impl Drop for TypeTree {
        fn drop(&mut self) {
            unsafe { EnzymeFreeTypeTree(self.inner) }
        }
    }

    #[repr(C)]
    #[derive(Debug, Copy, Clone)]
    pub struct CFnTypeInfo {
        #[doc = " Types of arguments, assumed of size len(Arguments)"]
        pub Arguments: *mut CTypeTreeRef,
        #[doc = " Type of return"]
        pub Return: CTypeTreeRef,
        #[doc = " The specific constant(s) known to represented by an argument, if constant"]
        pub KnownValues: *mut IntList,
    }
}

#[cfg(llvm_enzyme)]
pub use self::Enzyme_AD::*;

// Enzyme is an optional component, so we do need to provide a fallback when it is ont getting
// compiled. We deny the usage of #[autodiff(..)] on a higher level, so a placeholder implementation
// here is completely fine.
#[cfg(llvm_enzyme)]
pub mod Enzyme_AD {
    use libc::{c_char, c_void, size_t};

    use super::*;

    extern "C" {
        pub fn EnzymeNewTypeTree() -> CTypeTreeRef;
        pub fn EnzymeFreeTypeTree(CTT: CTypeTreeRef);
        pub fn EnzymeSetCLBool(arg1: *mut ::std::os::raw::c_void, arg2: u8);
        pub fn EnzymeSetCLInteger(arg1: *mut ::std::os::raw::c_void, arg2: i64);
    }

    extern "C" {
        static mut MaxIntOffset: c_void;
        static mut MaxTypeOffset: c_void;
        static mut EnzymeMaxTypeDepth: c_void;

        static mut EnzymeRuntimeActivityCheck: c_void;
        static mut EnzymePrintPerf: c_void;
        static mut EnzymePrintActivity: c_void;
        static mut EnzymePrintType: c_void;
        static mut EnzymePrint: c_void;
        static mut EnzymeStrictAliasing: c_void;
        static mut looseTypeAnalysis: c_void;
        static mut EnzymeInline: c_void;
    }
    pub fn set_runtime_activity_check(check: bool) {
        unsafe {
            EnzymeSetCLBool(std::ptr::addr_of_mut!(EnzymeRuntimeActivityCheck), check as u8);
        }
    }
    pub fn set_max_int_offset(offset: u64) {
        let offset = offset.try_into().unwrap();
        unsafe {
            EnzymeSetCLInteger(std::ptr::addr_of_mut!(MaxIntOffset), offset);
        }
    }
    pub fn set_max_type_offset(offset: u64) {
        let offset = offset.try_into().unwrap();
        unsafe {
            EnzymeSetCLInteger(std::ptr::addr_of_mut!(MaxTypeOffset), offset);
        }
    }
    pub fn set_max_type_depth(depth: u64) {
        let depth = depth.try_into().unwrap();
        unsafe {
            EnzymeSetCLInteger(std::ptr::addr_of_mut!(EnzymeMaxTypeDepth), depth);
        }
    }
    pub fn set_print_perf(print: bool) {
        unsafe {
            EnzymeSetCLBool(std::ptr::addr_of_mut!(EnzymePrintPerf), print as u8);
        }
    }
    pub fn set_print_activity(print: bool) {
        unsafe {
            EnzymeSetCLBool(std::ptr::addr_of_mut!(EnzymePrintActivity), print as u8);
        }
    }
    pub fn set_print_type(print: bool) {
        unsafe {
            EnzymeSetCLBool(std::ptr::addr_of_mut!(EnzymePrintType), print as u8);
        }
    }
    pub fn set_print(print: bool) {
        unsafe {
            EnzymeSetCLBool(std::ptr::addr_of_mut!(EnzymePrint), print as u8);
        }
    }
    pub fn set_strict_aliasing(strict: bool) {
        unsafe {
            EnzymeSetCLBool(std::ptr::addr_of_mut!(EnzymeStrictAliasing), strict as u8);
        }
    }
    pub fn set_loose_types(loose: bool) {
        unsafe {
            EnzymeSetCLBool(std::ptr::addr_of_mut!(looseTypeAnalysis), loose as u8);
        }
    }
    pub fn set_inline(val: bool) {
        unsafe {
            EnzymeSetCLBool(std::ptr::addr_of_mut!(EnzymeInline), val as u8);
        }
    }

    extern "C" {
        pub fn EnzymeCreatePrimalAndGradient<'a>(
            arg1: EnzymeLogicRef,
            _builderCtx: *const u8, // &'a Builder<'_>,
            _callerCtx: *const u8,  // &'a Value,
            todiff: &'a Value,
            retType: CDIFFE_TYPE,
            constant_args: *const CDIFFE_TYPE,
            constant_args_size: size_t,
            TA: EnzymeTypeAnalysisRef,
            returnValue: u8,
            dretUsed: u8,
            mode: CDerivativeMode,
            width: ::std::os::raw::c_uint,
            freeMemory: u8,
            additionalArg: Option<&Type>,
            forceAnonymousTape: u8,
            typeInfo: CFnTypeInfo,
            _uncacheable_args: *const u8,
            uncacheable_args_size: size_t,
            augmented: EnzymeAugmentedReturnPtr,
            AtomicAdd: u8,
        ) -> &'a Value;
    }
    extern "C" {
        pub fn EnzymeCreateForwardDiff<'a>(
            arg1: EnzymeLogicRef,
            _builderCtx: *const u8, // &'a Builder<'_>,
            _callerCtx: *const u8,  // &'a Value,
            todiff: &'a Value,
            retType: CDIFFE_TYPE,
            constant_args: *const CDIFFE_TYPE,
            constant_args_size: size_t,
            TA: EnzymeTypeAnalysisRef,
            returnValue: u8,
            mode: CDerivativeMode,
            freeMemory: u8,
            width: ::std::os::raw::c_uint,
            additionalArg: Option<&Type>,
            typeInfo: CFnTypeInfo,
            _uncacheable_args: *const u8,
            uncacheable_args_size: size_t,
            augmented: EnzymeAugmentedReturnPtr,
        ) -> &'a Value;
    }
    pub type CustomRuleType = ::std::option::Option<
        unsafe extern "C" fn(
            direction: ::std::os::raw::c_int,
            ret: CTypeTreeRef,
            args: *mut CTypeTreeRef,
            known_values: *mut IntList,
            num_args: size_t,
            fnc: &Value,
            ta: *const ::std::os::raw::c_void,
        ) -> u8,
    >;
    extern "C" {
        pub fn CreateTypeAnalysis(
            Log: EnzymeLogicRef,
            customRuleNames: *mut *mut ::std::os::raw::c_char,
            customRules: *mut CustomRuleType,
            numRules: size_t,
        ) -> EnzymeTypeAnalysisRef;
    }
    extern "C" {
        //pub(super) fn ClearTypeAnalysis(arg1: EnzymeTypeAnalysisRef);
        pub fn FreeTypeAnalysis(arg1: EnzymeTypeAnalysisRef);
        pub fn CreateEnzymeLogic(PostOpt: u8) -> EnzymeLogicRef;
        pub fn ClearEnzymeLogic(arg1: EnzymeLogicRef);
        pub fn FreeEnzymeLogic(arg1: EnzymeLogicRef);
    }

    extern "C" {
        pub(super) fn EnzymeNewTypeTreeCT(arg1: CConcreteType, ctx: &Context) -> CTypeTreeRef;
        pub(super) fn EnzymeNewTypeTreeTR(arg1: CTypeTreeRef) -> CTypeTreeRef;
        pub(super) fn EnzymeMergeTypeTree(arg1: CTypeTreeRef, arg2: CTypeTreeRef) -> bool;
        pub(super) fn EnzymeTypeTreeOnlyEq(arg1: CTypeTreeRef, pos: i64);
        pub(super) fn EnzymeTypeTreeData0Eq(arg1: CTypeTreeRef);
        pub(super) fn EnzymeTypeTreeShiftIndiciesEq(
            arg1: CTypeTreeRef,
            data_layout: *const c_char,
            offset: i64,
            max_size: i64,
            add_offset: u64,
        );
        pub fn EnzymeTypeTreeToStringFree(arg1: *const c_char);
        pub fn EnzymeTypeTreeToString(arg1: CTypeTreeRef) -> *const c_char;
    }
}
