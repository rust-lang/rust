#![expect(dead_code)]

use libc::{c_char, c_uint};

use super::MetadataKindId;
use super::ffi::{AttributeKind, BasicBlock, Context, Metadata, Module, Type, Value};
use crate::llvm::{Bool, Builder};

// TypeTree types
pub(crate) type CTypeTreeRef = *mut EnzymeTypeTree;

#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub(crate) struct EnzymeTypeTree {
    _unused: [u8; 0],
}

#[repr(u32)]
#[derive(Debug, Copy, Clone, Hash, PartialEq, Eq)]
#[allow(non_camel_case_types)]
pub(crate) enum CConcreteType {
    DT_Anything = 0,
    DT_Integer = 1,
    DT_Pointer = 2,
    DT_Half = 3,
    DT_Float = 4,
    DT_Double = 5,
    DT_Unknown = 6,
    DT_FP128 = 9,
}

pub(crate) struct TypeTree {
    pub(crate) inner: CTypeTreeRef,
}

#[link(name = "llvm-wrapper", kind = "static")]
unsafe extern "C" {
    // Enzyme
    pub(crate) safe fn LLVMRustHasMetadata(I: &Value, KindID: MetadataKindId) -> bool;
    pub(crate) fn LLVMRustEraseInstUntilInclusive(BB: &BasicBlock, I: &Value);
    pub(crate) fn LLVMRustGetLastInstruction<'a>(BB: &BasicBlock) -> Option<&'a Value>;
    pub(crate) fn LLVMRustDIGetInstMetadata(I: &Value) -> Option<&Metadata>;
    pub(crate) fn LLVMRustEraseInstFromParent(V: &Value);
    pub(crate) fn LLVMRustGetTerminator<'a>(B: &BasicBlock) -> &'a Value;
    pub(crate) fn LLVMRustVerifyFunction(V: &Value, action: LLVMRustVerifierFailureAction) -> Bool;
    pub(crate) fn LLVMRustHasAttributeAtIndex(V: &Value, i: c_uint, Kind: AttributeKind) -> bool;
    pub(crate) fn LLVMRustGetArrayNumElements(Ty: &Type) -> u64;
    pub(crate) fn LLVMRustHasFnAttribute(
        F: &Value,
        Name: *const c_char,
        NameLen: libc::size_t,
    ) -> bool;
    pub(crate) fn LLVMRustRemoveFnAttribute(F: &Value, Name: *const c_char, NameLen: libc::size_t);
    pub(crate) fn LLVMGetFirstFunction(M: &Module) -> Option<&Value>;
    pub(crate) fn LLVMGetNextFunction(Fn: &Value) -> Option<&Value>;
    pub(crate) fn LLVMRustRemoveEnumAttributeAtIndex(
        Fn: &Value,
        index: c_uint,
        kind: AttributeKind,
    );
    pub(crate) fn LLVMRustPositionBefore<'a>(B: &'a Builder<'_>, I: &'a Value);
    pub(crate) fn LLVMRustPositionAfter<'a>(B: &'a Builder<'_>, I: &'a Value);
    pub(crate) fn LLVMRustGetFunctionCall(
        F: &Value,
        name: *const c_char,
        NameLen: libc::size_t,
    ) -> Option<&Value>;

}

unsafe extern "C" {
    // Enzyme
    pub(crate) fn LLVMDumpModule(M: &Module);
    pub(crate) fn LLVMDumpValue(V: &Value);
    pub(crate) fn LLVMGetFunctionCallConv(F: &Value) -> c_uint;
    pub(crate) fn LLVMGetReturnType(T: &Type) -> &Type;
    pub(crate) fn LLVMGetParams(Fnc: &Value, params: *mut &Value);
    pub(crate) fn LLVMGetNamedFunction(M: &Module, Name: *const c_char) -> Option<&Value>;
}

#[repr(C)]
#[derive(Copy, Clone, PartialEq)]
pub(crate) enum LLVMRustVerifierFailureAction {
    LLVMAbortProcessAction = 0,
    LLVMPrintMessageAction = 1,
    LLVMReturnStatusAction = 2,
}

#[cfg(feature = "llvm_enzyme")]
pub(crate) use self::Enzyme_AD::*;

#[cfg(feature = "llvm_enzyme")]
pub(crate) mod Enzyme_AD {
    use std::ffi::{c_char, c_void};
    use std::sync::{Mutex, MutexGuard, OnceLock};

    use rustc_middle::bug;
    use rustc_session::config::{Sysroot, host_tuple};
    use rustc_session::filesearch;

    use super::{CConcreteType, CTypeTreeRef, Context};
    use crate::llvm::{EnzymeTypeTree, LLVMRustVersionMajor};

    type EnzymeSetCLBoolFn = unsafe extern "C" fn(*mut c_void, u8);
    type EnzymeSetCLStringFn = unsafe extern "C" fn(*mut c_void, *const c_char);

    type EnzymeNewTypeTreeFn = unsafe extern "C" fn() -> CTypeTreeRef;
    type EnzymeNewTypeTreeCTFn = unsafe extern "C" fn(CConcreteType, &Context) -> CTypeTreeRef;
    type EnzymeNewTypeTreeTRFn = unsafe extern "C" fn(CTypeTreeRef) -> CTypeTreeRef;
    type EnzymeFreeTypeTreeFn = unsafe extern "C" fn(CTypeTreeRef);
    type EnzymeMergeTypeTreeFn = unsafe extern "C" fn(CTypeTreeRef, CTypeTreeRef) -> bool;
    type EnzymeTypeTreeOnlyEqFn = unsafe extern "C" fn(CTypeTreeRef, i64);
    type EnzymeTypeTreeData0EqFn = unsafe extern "C" fn(CTypeTreeRef);
    type EnzymeTypeTreeShiftIndiciesEqFn =
        unsafe extern "C" fn(CTypeTreeRef, *const c_char, i64, i64, u64);
    type EnzymeTypeTreeInsertEqFn =
        unsafe extern "C" fn(CTypeTreeRef, *const i64, usize, CConcreteType, &Context);
    type EnzymeTypeTreeToStringFn = unsafe extern "C" fn(CTypeTreeRef) -> *const c_char;
    type EnzymeTypeTreeToStringFreeFn = unsafe extern "C" fn(*const c_char);

    #[allow(non_snake_case)]
    pub(crate) struct EnzymeWrapper {
        EnzymeNewTypeTree: EnzymeNewTypeTreeFn,
        EnzymeNewTypeTreeCT: EnzymeNewTypeTreeCTFn,
        EnzymeNewTypeTreeTR: EnzymeNewTypeTreeTRFn,
        EnzymeFreeTypeTree: EnzymeFreeTypeTreeFn,
        EnzymeMergeTypeTree: EnzymeMergeTypeTreeFn,
        EnzymeTypeTreeOnlyEq: EnzymeTypeTreeOnlyEqFn,
        EnzymeTypeTreeData0Eq: EnzymeTypeTreeData0EqFn,
        EnzymeTypeTreeShiftIndiciesEq: EnzymeTypeTreeShiftIndiciesEqFn,
        EnzymeTypeTreeInsertEq: EnzymeTypeTreeInsertEqFn,
        EnzymeTypeTreeToString: EnzymeTypeTreeToStringFn,
        EnzymeTypeTreeToStringFree: EnzymeTypeTreeToStringFreeFn,

        EnzymePrintPerf: *mut c_void,
        EnzymePrintActivity: *mut c_void,
        EnzymePrintType: *mut c_void,
        EnzymeFunctionToAnalyze: *mut c_void,
        EnzymePrint: *mut c_void,
        EnzymeStrictAliasing: *mut c_void,
        EnzymeInline: *mut c_void,
        EnzymeMaxTypeDepth: *mut c_void,
        RustTypeRules: *mut c_void,
        looseTypeAnalysis: *mut c_void,

        EnzymeSetCLBool: EnzymeSetCLBoolFn,
        EnzymeSetCLString: EnzymeSetCLStringFn,
        pub registerEnzymeAndPassPipeline: *const c_void,
        lib: libloading::Library,
    }

    unsafe impl Sync for EnzymeWrapper {}
    unsafe impl Send for EnzymeWrapper {}

    fn load_ptr_by_symbol_mut_void(
        lib: &libloading::Library,
        bytes: &[u8],
    ) -> Result<*mut c_void, Box<dyn std::error::Error>> {
        unsafe {
            let s: libloading::Symbol<'_, *mut c_void> = lib.get(bytes)?;
            // libloading = 0.9.0: try_as_raw_ptr always succeeds and returns Some
            let s = s.try_as_raw_ptr().unwrap();
            Ok(s)
        }
    }

    // e.g.
    // load_ptrs_by_symbols_mut_void(ABC, XYZ);
    // =>
    // let ABC = load_ptr_mut_void(&lib, b"ABC")?;
    // let XYZ = load_ptr_mut_void(&lib, b"XYZ")?;
    macro_rules! load_ptrs_by_symbols_mut_void {
        ($lib:expr, $($name:ident),* $(,)?) => {
            $(
                #[allow(non_snake_case)]
                let $name = load_ptr_by_symbol_mut_void(&$lib, stringify!($name).as_bytes())?;
            )*
        };
    }

    // e.g.
    // load_ptrs_by_symbols_fn(ABC: ABCFn, XYZ: XYZFn);
    // =>
    // let ABC: libloading::Symbol<'_, ABCFn> = unsafe { lib.get(b"ABC")? };
    // let XYZ: libloading::Symbol<'_, XYZFn> = unsafe { lib.get(b"XYZ")? };
    macro_rules! load_ptrs_by_symbols_fn {
        ($lib:expr, $($name:ident : $ty:ty),* $(,)?) => {
            $(
                #[allow(non_snake_case)]
                let $name: $ty = *unsafe { $lib.get::<$ty>(stringify!($name).as_bytes())? };
            )*
        };
    }

    static ENZYME_INSTANCE: OnceLock<Mutex<EnzymeWrapper>> = OnceLock::new();

    impl EnzymeWrapper {
        /// Initialize EnzymeWrapper with the given sysroot if not already initialized.
        /// Safe to call multiple times - subsequent calls are no-ops due to OnceLock.
        pub(crate) fn get_or_init(
            sysroot: &rustc_session::config::Sysroot,
        ) -> Result<MutexGuard<'static, Self>, Box<dyn std::error::Error>> {
            let mtx: &'static Mutex<EnzymeWrapper> = ENZYME_INSTANCE.get_or_try_init(|| {
                let w = Self::call_dynamic(sysroot)?;
                Ok::<_, Box<dyn std::error::Error>>(Mutex::new(w))
            })?;

            Ok(mtx.lock().unwrap())
        }

        /// Get the EnzymeWrapper instance. Panics if not initialized.
        pub(crate) fn get_instance() -> MutexGuard<'static, Self> {
            ENZYME_INSTANCE
                .get()
                .expect("EnzymeWrapper not initialized. Call get_or_init with sysroot first.")
                .lock()
                .unwrap()
        }

        pub(crate) fn new_type_tree(&self) -> CTypeTreeRef {
            unsafe { (self.EnzymeNewTypeTree)() }
        }

        pub(crate) fn new_type_tree_ct(
            &self,
            t: CConcreteType,
            ctx: &Context,
        ) -> *mut EnzymeTypeTree {
            unsafe { (self.EnzymeNewTypeTreeCT)(t, ctx) }
        }

        pub(crate) fn new_type_tree_tr(&self, tree: CTypeTreeRef) -> CTypeTreeRef {
            unsafe { (self.EnzymeNewTypeTreeTR)(tree) }
        }

        pub(crate) fn free_type_tree(&self, tree: CTypeTreeRef) {
            unsafe { (self.EnzymeFreeTypeTree)(tree) }
        }

        pub(crate) fn merge_type_tree(&self, tree1: CTypeTreeRef, tree2: CTypeTreeRef) -> bool {
            unsafe { (self.EnzymeMergeTypeTree)(tree1, tree2) }
        }

        pub(crate) fn tree_only_eq(&self, tree: CTypeTreeRef, num: i64) {
            unsafe { (self.EnzymeTypeTreeOnlyEq)(tree, num) }
        }

        pub(crate) fn tree_data0_eq(&self, tree: CTypeTreeRef) {
            unsafe { (self.EnzymeTypeTreeData0Eq)(tree) }
        }

        pub(crate) fn shift_indicies_eq(
            &self,
            tree: CTypeTreeRef,
            data_layout: *const c_char,
            offset: i64,
            max_size: i64,
            add_offset: u64,
        ) {
            unsafe {
                (self.EnzymeTypeTreeShiftIndiciesEq)(
                    tree,
                    data_layout,
                    offset,
                    max_size,
                    add_offset,
                )
            }
        }

        pub(crate) fn tree_insert_eq(
            &self,
            tree: CTypeTreeRef,
            indices: *const i64,
            len: usize,
            ct: CConcreteType,
            ctx: &Context,
        ) {
            unsafe { (self.EnzymeTypeTreeInsertEq)(tree, indices, len, ct, ctx) }
        }

        pub(crate) fn tree_to_string(&self, tree: *mut EnzymeTypeTree) -> *const c_char {
            unsafe { (self.EnzymeTypeTreeToString)(tree) }
        }

        pub(crate) fn tree_to_string_free(&self, ch: *const c_char) {
            unsafe { (self.EnzymeTypeTreeToStringFree)(ch) }
        }

        pub(crate) fn get_max_type_depth(&self) -> usize {
            unsafe { std::ptr::read::<u32>(self.EnzymeMaxTypeDepth as *const u32) as usize }
        }

        pub(crate) fn set_print_perf(&mut self, print: bool) {
            unsafe {
                (self.EnzymeSetCLBool)(self.EnzymePrintPerf, print as u8);
            }
        }

        pub(crate) fn set_print_activity(&mut self, print: bool) {
            unsafe {
                (self.EnzymeSetCLBool)(self.EnzymePrintActivity, print as u8);
            }
        }

        pub(crate) fn set_print_type(&mut self, print: bool) {
            unsafe {
                (self.EnzymeSetCLBool)(self.EnzymePrintType, print as u8);
            }
        }

        pub(crate) fn set_print_type_fun(&mut self, fun_name: &str) {
            let c_fun_name = std::ffi::CString::new(fun_name)
                .unwrap_or_else(|err| bug!("failed to set_print_type_fun: {err}"));
            unsafe {
                (self.EnzymeSetCLString)(
                    self.EnzymeFunctionToAnalyze,
                    c_fun_name.as_ptr() as *const c_char,
                );
            }
        }

        pub(crate) fn set_print(&mut self, print: bool) {
            unsafe {
                (self.EnzymeSetCLBool)(self.EnzymePrint, print as u8);
            }
        }

        pub(crate) fn set_strict_aliasing(&mut self, strict: bool) {
            unsafe {
                (self.EnzymeSetCLBool)(self.EnzymeStrictAliasing, strict as u8);
            }
        }

        pub(crate) fn set_loose_types(&mut self, loose: bool) {
            unsafe {
                (self.EnzymeSetCLBool)(self.looseTypeAnalysis, loose as u8);
            }
        }

        pub(crate) fn set_inline(&mut self, val: bool) {
            unsafe {
                (self.EnzymeSetCLBool)(self.EnzymeInline, val as u8);
            }
        }

        pub(crate) fn set_rust_rules(&mut self, val: bool) {
            unsafe {
                (self.EnzymeSetCLBool)(self.RustTypeRules, val as u8);
            }
        }

        #[allow(non_snake_case)]
        fn call_dynamic(
            sysroot: &rustc_session::config::Sysroot,
        ) -> Result<Self, Box<dyn std::error::Error>> {
            let enzyme_path = Self::get_enzyme_path(sysroot)?;
            let lib = unsafe { libloading::Library::new(enzyme_path)? };

            load_ptrs_by_symbols_fn!(
                lib,
                EnzymeNewTypeTree: EnzymeNewTypeTreeFn,
                EnzymeNewTypeTreeCT: EnzymeNewTypeTreeCTFn,
                EnzymeNewTypeTreeTR: EnzymeNewTypeTreeTRFn,
                EnzymeFreeTypeTree: EnzymeFreeTypeTreeFn,
                EnzymeMergeTypeTree: EnzymeMergeTypeTreeFn,
                EnzymeTypeTreeOnlyEq: EnzymeTypeTreeOnlyEqFn,
                EnzymeTypeTreeData0Eq: EnzymeTypeTreeData0EqFn,
                EnzymeTypeTreeShiftIndiciesEq: EnzymeTypeTreeShiftIndiciesEqFn,
                EnzymeTypeTreeInsertEq: EnzymeTypeTreeInsertEqFn,
                EnzymeTypeTreeToString: EnzymeTypeTreeToStringFn,
                EnzymeTypeTreeToStringFree: EnzymeTypeTreeToStringFreeFn,
                EnzymeSetCLBool: EnzymeSetCLBoolFn,
                EnzymeSetCLString: EnzymeSetCLStringFn,
            );

            load_ptrs_by_symbols_mut_void!(
                lib,
                registerEnzymeAndPassPipeline,
                EnzymePrintPerf,
                EnzymePrintActivity,
                EnzymePrintType,
                EnzymeFunctionToAnalyze,
                EnzymePrint,
                EnzymeStrictAliasing,
                EnzymeInline,
                EnzymeMaxTypeDepth,
                RustTypeRules,
                looseTypeAnalysis,
            );

            Ok(Self {
                EnzymeNewTypeTree,
                EnzymeNewTypeTreeCT,
                EnzymeNewTypeTreeTR,
                EnzymeFreeTypeTree,
                EnzymeMergeTypeTree,
                EnzymeTypeTreeOnlyEq,
                EnzymeTypeTreeData0Eq,
                EnzymeTypeTreeShiftIndiciesEq,
                EnzymeTypeTreeInsertEq,
                EnzymeTypeTreeToString,
                EnzymeTypeTreeToStringFree,
                EnzymePrintPerf,
                EnzymePrintActivity,
                EnzymePrintType,
                EnzymeFunctionToAnalyze,
                EnzymePrint,
                EnzymeStrictAliasing,
                EnzymeInline,
                EnzymeMaxTypeDepth,
                RustTypeRules,
                looseTypeAnalysis,
                EnzymeSetCLBool,
                EnzymeSetCLString,
                registerEnzymeAndPassPipeline,
                lib,
            })
        }

        fn get_enzyme_path(sysroot: &Sysroot) -> Result<String, String> {
            let llvm_version_major = unsafe { LLVMRustVersionMajor() };

            let path_buf = sysroot
                .all_paths()
                .map(|sysroot_path| {
                    filesearch::make_target_lib_path(sysroot_path, host_tuple())
                        .join("lib")
                        .with_file_name(format!("libEnzyme-{llvm_version_major}"))
                        .with_extension(std::env::consts::DLL_EXTENSION)
                })
                .find(|f| f.exists())
                .ok_or_else(|| {
                    let candidates = sysroot
                        .all_paths()
                        .map(|p| p.join("lib").display().to_string())
                        .collect::<Vec<String>>()
                        .join("\n* ");
                    format!(
                        "failed to find a `libEnzyme-{llvm_version_major}` folder \
                    in the sysroot candidates:\n* {candidates}"
                    )
                })?;

            Ok(path_buf
                .to_str()
                .ok_or_else(|| format!("invalid UTF-8 in path: {}", path_buf.display()))?
                .to_string())
        }
    }
}

#[cfg(not(feature = "llvm_enzyme"))]
pub(crate) use self::Fallback_AD::*;

#[cfg(not(feature = "llvm_enzyme"))]
pub(crate) mod Fallback_AD {
    #![allow(unused_variables)]

    use std::ffi::c_void;
    use std::sync::{Mutex, MutexGuard};

    use libc::c_char;
    use rustc_codegen_ssa::back::write::CodegenContext;
    use rustc_codegen_ssa::traits::WriteBackendMethods;

    use super::{CConcreteType, CTypeTreeRef, Context, EnzymeTypeTree};

    pub(crate) struct EnzymeWrapper {
        pub registerEnzymeAndPassPipeline: *const c_void,
    }

    impl EnzymeWrapper {
        pub(crate) fn get_or_init(
            _sysroot: &rustc_session::config::Sysroot,
        ) -> Result<MutexGuard<'static, Self>, Box<dyn std::error::Error>> {
            unimplemented!("Enzyme not available: build with llvm_enzyme feature")
        }

        pub(crate) fn init<'a, B: WriteBackendMethods>(
            _cgcx: &'a CodegenContext<B>,
        ) -> &'static Mutex<Self> {
            unimplemented!("Enzyme not available: build with llvm_enzyme feature")
        }

        pub(crate) fn get_instance() -> MutexGuard<'static, Self> {
            unimplemented!("Enzyme not available: build with llvm_enzyme feature")
        }

        pub(crate) fn new_type_tree(&self) -> CTypeTreeRef {
            unimplemented!()
        }

        pub(crate) fn new_type_tree_ct(
            &self,
            t: CConcreteType,
            ctx: &Context,
        ) -> *mut EnzymeTypeTree {
            unimplemented!()
        }

        pub(crate) fn new_type_tree_tr(&self, tree: CTypeTreeRef) -> CTypeTreeRef {
            unimplemented!()
        }

        pub(crate) fn free_type_tree(&self, tree: CTypeTreeRef) {
            unimplemented!()
        }

        pub(crate) fn merge_type_tree(&self, tree1: CTypeTreeRef, tree2: CTypeTreeRef) -> bool {
            unimplemented!()
        }

        pub(crate) fn tree_only_eq(&self, tree: CTypeTreeRef, num: i64) {
            unimplemented!()
        }

        pub(crate) fn tree_data0_eq(&self, tree: CTypeTreeRef) {
            unimplemented!()
        }

        pub(crate) fn shift_indicies_eq(
            &self,
            tree: CTypeTreeRef,
            data_layout: *const c_char,
            offset: i64,
            max_size: i64,
            add_offset: u64,
        ) {
            unimplemented!()
        }

        pub(crate) fn tree_insert_eq(
            &self,
            tree: CTypeTreeRef,
            indices: *const i64,
            len: usize,
            ct: CConcreteType,
            ctx: &Context,
        ) {
            unimplemented!()
        }

        pub(crate) fn tree_to_string(&self, tree: *mut EnzymeTypeTree) -> *const c_char {
            unimplemented!()
        }

        pub(crate) fn tree_to_string_free(&self, ch: *const c_char) {
            unimplemented!()
        }

        pub(crate) fn get_max_type_depth(&self) -> usize {
            unimplemented!()
        }

        pub(crate) fn set_inline(&mut self, val: bool) {
            unimplemented!()
        }

        pub(crate) fn set_print_perf(&mut self, print: bool) {
            unimplemented!()
        }

        pub(crate) fn set_print_activity(&mut self, print: bool) {
            unimplemented!()
        }

        pub(crate) fn set_print_type(&mut self, print: bool) {
            unimplemented!()
        }

        pub(crate) fn set_print_type_fun(&mut self, fun_name: &str) {
            unimplemented!()
        }

        pub(crate) fn set_print(&mut self, print: bool) {
            unimplemented!()
        }

        pub(crate) fn set_strict_aliasing(&mut self, strict: bool) {
            unimplemented!()
        }

        pub(crate) fn set_loose_types(&mut self, loose: bool) {
            unimplemented!()
        }

        pub(crate) fn set_rust_rules(&mut self, val: bool) {
            unimplemented!()
        }
    }
}

impl TypeTree {
    pub(crate) fn new() -> TypeTree {
        let wrapper = EnzymeWrapper::get_instance();
        let inner = wrapper.new_type_tree();
        TypeTree { inner }
    }

    pub(crate) fn from_type(t: CConcreteType, ctx: &Context) -> TypeTree {
        let wrapper = EnzymeWrapper::get_instance();
        let inner = wrapper.new_type_tree_ct(t, ctx);
        TypeTree { inner }
    }

    pub(crate) fn merge(self, other: Self) -> Self {
        let wrapper = EnzymeWrapper::get_instance();
        wrapper.merge_type_tree(self.inner, other.inner);
        drop(other);
        self
    }

    #[must_use]
    pub(crate) fn shift(
        self,
        layout: &str,
        offset: isize,
        max_size: isize,
        add_offset: usize,
    ) -> Self {
        let layout = std::ffi::CString::new(layout).unwrap();
        let wrapper = EnzymeWrapper::get_instance();
        wrapper.shift_indicies_eq(
            self.inner,
            layout.as_ptr(),
            offset as i64,
            max_size as i64,
            add_offset as u64,
        );

        self
    }

    pub(crate) fn insert(&mut self, indices: &[i64], ct: CConcreteType, ctx: &Context) {
        let wrapper = EnzymeWrapper::get_instance();
        wrapper.tree_insert_eq(self.inner, indices.as_ptr(), indices.len(), ct, ctx);
    }
}

impl Clone for TypeTree {
    fn clone(&self) -> Self {
        let wrapper = EnzymeWrapper::get_instance();
        let inner = wrapper.new_type_tree_tr(self.inner);
        TypeTree { inner }
    }
}

impl std::fmt::Display for TypeTree {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let wrapper = EnzymeWrapper::get_instance();
        let ptr = wrapper.tree_to_string(self.inner);
        let cstr = unsafe { std::ffi::CStr::from_ptr(ptr) };
        match cstr.to_str() {
            Ok(x) => write!(f, "{}", x)?,
            Err(err) => write!(f, "could not parse: {}", err)?,
        }

        // delete C string pointer
        wrapper.tree_to_string_free(ptr);

        Ok(())
    }
}

impl std::fmt::Debug for TypeTree {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        <Self as std::fmt::Display>::fmt(self, f)
    }
}

impl Drop for TypeTree {
    fn drop(&mut self) {
        let wrapper = EnzymeWrapper::get_instance();
        wrapper.free_type_tree(self.inner)
    }
}
