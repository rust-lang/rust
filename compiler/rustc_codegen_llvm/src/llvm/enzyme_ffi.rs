#![expect(dead_code)]


use tracing::info;

use std::path::Path;

use libc::{c_char, c_uint};

use super::MetadataKindId;
use super::ffi::{AttributeKind, BasicBlock, Metadata, Module, Type, Value};
use crate::llvm::{Bool, Builder};

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

#[cfg(llvm_enzyme)]
pub(crate) use self::Enzyme_AD::*;

//#[cfg(llvm_enzyme)]
pub(crate) mod Enzyme_AD {
//    use std::ffi::CString;
//
//    use libc::c_void;
//
//type SetFlag = unsafe extern "C" fn(*mut c_void, u8);
//
//#[derive(Debug)]
//pub(crate) struct EnzymeFns {
//    pub set_cl: SetFlag,
//}
//
//#[derive(Debug)]
//pub(crate) struct EnzymeWrapper {
//    EnzymePrintPerf: *mut c_void,
//    EnzymePrintActivity: *mut c_void,
//    EnzymePrintType: *mut c_void,
//    EnzymeFunctionToAnalyze: *mut c_void,
//    EnzymePrint: *mut c_void,
//    EnzymeStrictAliasing: *mut c_void,
//    looseTypeAnalysis: *mut c_void,
//    EnzymeInline: *mut c_void,
//    RustTypeRules: *mut c_void,
//
//    EnzymeSetCLBool: EnzymeFns,
//    EnzymeSetCLString: EnzymeFns,
//    pub registerEnzymeAndPassPipeline: *const c_void,
//    lib: libloading::Library,
//}
//        fn call_dynamic() -> Result<EnzymeWrapper, Box<dyn std::error::Error>> {
//            fn load_ptr(lib: &libloading::Library, bytes: &[u8]) -> Result<*mut c_void, Box<dyn std::error::Error>> {
//                // Safety: symbol lookup from a loaded shared object.
//                unsafe {
//                    let s: libloading::Symbol<'_, *mut c_void> = lib.get(bytes)?;
//                    let s = s.try_as_raw_ptr().unwrap();
//                    Ok(s as *mut c_void)
//                }
//            }
//            dbg!("starting");
//            dbg!("Loading Enzyme");
//            use std::sync::OnceLock;
//            static ENZYME_PATH: OnceLock<String> = OnceLock::new();
//            assert!(ENZYME_PATH.get().is_some());
//            let mypath = ENZYME_PATH.get().unwrap(); // load Library from mypath
//            let lib = unsafe {libloading::Library::new(mypath)?};
//            //let lib = unsafe {libloading::Library::new("/home/manuel/prog/rust/build/x86_64-unknown-linux-gnu/enzyme/lib/libEnzyme-21.so")?};
//            dbg!("second");
//            let EnzymeSetCLBool: libloading::Symbol<'_, SetFlag> = unsafe{lib.get(b"EnzymeSetCLBool")?};
//            dbg!("third");
//            let registerEnzymeAndPassPipeline =
//                load_ptr(&lib, b"registerEnzymeAndPassPipeline").unwrap() as *const c_void;
//            dbg!("fourth");
//            let EnzymeSetCLString: libloading::Symbol<'_, SetFlag> = unsafe{ lib.get(b"EnzymeSetCLString")?};
//            dbg!("done");
//
//            let EnzymePrintPerf = load_ptr(&lib, b"EnzymePrintPerf").unwrap();
//            let EnzymePrintActivity = load_ptr(&lib, b"EnzymePrintActivity").unwrap();
//            let EnzymePrintType = load_ptr(&lib, b"EnzymePrintType").unwrap();
//            let EnzymeFunctionToAnalyze = load_ptr(&lib, b"EnzymeFunctionToAnalyze").unwrap();
//            let EnzymePrint = load_ptr(&lib, b"EnzymePrint").unwrap();
//
//            let EnzymeStrictAliasing = load_ptr(&lib, b"EnzymeStrictAliasing").unwrap();
//            let looseTypeAnalysis = load_ptr(&lib, b"looseTypeAnalysis").unwrap();
//            let EnzymeInline = load_ptr(&lib, b"EnzymeInline").unwrap();
//            let RustTypeRules = load_ptr(&lib, b"RustTypeRules").unwrap();
//
//            let wrap = EnzymeWrapper {
//                EnzymePrintPerf,
//                EnzymePrintActivity,
//                EnzymePrintType,
//                EnzymeFunctionToAnalyze,
//                EnzymePrint,
//                EnzymeStrictAliasing,
//                looseTypeAnalysis,
//                EnzymeInline,
//                RustTypeRules,
//                EnzymeSetCLBool: EnzymeFns {set_cl: *EnzymeSetCLBool},
//                EnzymeSetCLString: EnzymeFns {set_cl: *EnzymeSetCLString},
//                registerEnzymeAndPassPipeline,
//                lib
//            };
//            dbg!(&wrap);
//            Ok(wrap)
//        }
use std::sync::Mutex;
use rustc_middle::bug;
use tracing::info;
use rustc_session::filesearch;
use rustc_session::Session;
use rustc_session::config::host_tuple;
//unsafe impl Sync for EnzymeWrapper {}
//unsafe impl Send for EnzymeWrapper {}
//    impl EnzymeWrapper {
//        pub(crate) fn current() -> &'static Mutex<EnzymeWrapper> {
//            use std::sync::OnceLock;
//            static CELL: OnceLock<Mutex<EnzymeWrapper>> = OnceLock::new();
//            static ENZYME_PATH: OnceLock<String> = OnceLock::new();
//            fn init_enzyme() -> Mutex<EnzymeWrapper> {
//                call_dynamic().unwrap().into()
//            }
//            //ENZYME_PATH.wait();
//            if ENZYME_PATH.get().is_none() {
//                bug!("enzyme path is none!");
//            }
//            CELL.get_or_init(|| init_enzyme())
//        }
//        pub(crate) fn set_path(session: &Session) -> String {
//            fn get_enzyme_path(session: &Session) -> String {
//                dbg!("starting");
//                dbg!("Loading Enzyme");
//                let target = host_tuple();
//                let lib_ext = std::env::consts::DLL_EXTENSION;
//                let sysroot = &session.opts.sysroot;
//                //dbg!(sysroot);
//
//                let sysroot = sysroot
//                    .all_paths()
//                    .map(|sysroot| {
//                        filesearch::make_target_lib_path(sysroot, target).join("lib").with_file_name("libEnzyme-21").with_extension(lib_ext)
//                        //filesearch::make_target_lib_path(sysroot, target).join("lib").with_file_name("lib")
//                    })
//                    .find(|f| {
//                        info!("Enzyme candidate: {}", f.display());
//                        f.exists()
//                    })
//                    .unwrap_or_else(|| {
//                        let candidates = sysroot
//                            .all_paths()
//                            .map(|p| p.join("lib").display().to_string())
//                            .collect::<Vec<_>>()
//                            .join("\n* ");
//                        let err = format!(
//                            "failed to find a `libEnzyme` folder \
//                                       in the sysroot candidates:\n* {candidates}"
//                        );
//                        dbg!(&err);
//                        bug!("asdf");
//                        //early_dcx.early_fatal(err);
//                    });
//
//                info!("probing {} for a codegen backend", sysroot.display());
//                let enzyme_path = sysroot.to_str().unwrap().to_string();
//                //dbg!(&enzyme_path);
//                enzyme_path
//            }
//            use std::sync::OnceLock;
//            static ENZYME_PATH: OnceLock<String> = OnceLock::new();
//            ENZYME_PATH.get_or_init(|| get_enzyme_path(session)).to_string()
//            //ENZYME_PATH.get().unwrap().to_string()
//            //ENZYME_PATH.get_or_init(|| get_enzyme_path(session)).clone()
//        }
//        pub(crate) fn set_print_perf(&mut self, print: bool) {
//            unsafe {
//                //(self.EnzymeSetCLBool.set_cl)(self.EnzymePrintPerf, print as u8);
//                //(self.EnzymeSetCLBool)(std::ptr::addr_of_mut!(self.EnzymePrintPerf), print as u8);
//            }
//        }
//
//        pub(crate) fn set_print_activity(&mut self, print: bool) {
//            unsafe {
//                //(self.EnzymeSetCLBool.set_cl)(self.EnzymePrintActivity, print as u8);
//                //(self.EnzymeSetCLBool)(std::ptr::addr_of_mut!(self.EnzymePrintActivity), print as u8);
//            }
//        }
//
//        pub(crate) fn set_print_type(&mut self, print: bool) {
//            unsafe {
//               // (self.EnzymeSetCLBool.set_cl)(self.EnzymePrintType, print as u8);
//            }
//        }
//
//        pub(crate) fn set_print_type_fun(&mut self, fun_name: &str) {
//            let _c_fun_name = CString::new(fun_name).unwrap();
//            //unsafe {
//            //    (self.EnzymeSetCLString.set_cl)(
//            //        self.EnzymeFunctionToAnalyze,
//            //        c_fun_name.as_ptr() as *const c_char,
//            //    );
//            //}
//        }
//
//        pub(crate) fn set_print(&mut self, print: bool) {
//            unsafe {
//                //(self.EnzymeSetCLBool.set_cl)(self.EnzymePrint, print as u8);
//            }
//        }
//
//        pub(crate) fn set_strict_aliasing(&mut self, strict: bool) {
//            unsafe {
//                //(self.EnzymeSetCLBool.set_cl)(self.EnzymeStrictAliasing, strict as u8);
//            }
//        }
//
//        pub(crate) fn set_loose_types(&mut self, loose: bool) {
//            unsafe {
//                //(self.EnzymeSetCLBool.set_cl)(self.looseTypeAnalysis, loose as u8);
//            }
//        }
//
//        pub(crate) fn set_inline(&mut self, val: bool) {
//            unsafe {
//                //(self.EnzymeSetCLBool.set_cl)(self.EnzymeInline, val as u8);
//            }
//        }
//
//        pub(crate) fn set_rust_rules(&mut self, val: bool) {
//            unsafe {
//                //(self.EnzymeSetCLBool.set_cl)(self.RustTypeRules, val as u8);
//            }
//        }
//    }


}

#[cfg(not(llvm_enzyme))]
pub(crate) use self::Fallback_AD::*;

#[cfg(not(llvm_enzyme))]
pub(crate) mod Fallback_AD {
    #![allow(unused_variables)]

    pub(crate) fn set_inline(val: bool) {
        unimplemented!()
    }
    pub(crate) fn set_print_perf(print: bool) {
        unimplemented!()
    }
    pub(crate) fn set_print_activity(print: bool) {
        unimplemented!()
    }
    pub(crate) fn set_print_type(print: bool) {
        unimplemented!()
    }
    pub(crate) fn set_print_type_fun(fun_name: &str) {
        unimplemented!()
    }
    pub(crate) fn set_print(print: bool) {
        unimplemented!()
    }
    pub(crate) fn set_strict_aliasing(strict: bool) {
        unimplemented!()
    }
    pub(crate) fn set_loose_types(loose: bool) {
        unimplemented!()
    }
    pub(crate) fn set_rust_rules(val: bool) {
        unimplemented!()
    }
}
