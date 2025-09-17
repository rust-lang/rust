#![expect(dead_code)]

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
    use std::ffi::CString;
    //use std::ffi::{CString, c_char};

    use libc::c_void;

type SetFlag = unsafe extern "C" fn(*mut c_void, u8);

#[derive(Debug)]
pub(crate) struct EnzymeFns {
    pub set_cl: SetFlag,
}

    #[derive(Debug)]
    pub(crate) struct EnzymeWrapper {
        EnzymePrintPerf: *mut c_void,
        EnzymePrintActivity: *mut c_void,
        EnzymePrintType: *mut c_void,
        EnzymeFunctionToAnalyze: *mut c_void,
        EnzymePrint: *mut c_void,
        EnzymeStrictAliasing: *mut c_void,
        looseTypeAnalysis: *mut c_void,
        EnzymeInline: *mut c_void,
        RustTypeRules: *mut c_void,

        EnzymeSetCLBool: EnzymeFns,
        EnzymeSetCLString: EnzymeFns,
        pub registerEnzymeAndPassPipeline: *const c_void,
    }
        fn call_dynamic() -> Result<EnzymeWrapper, Box<dyn std::error::Error>> {
            fn load_ptr(lib: &libloading::Library, bytes: &[u8]) -> Result<*mut c_void, Box<dyn std::error::Error>> {
                // Safety: symbol lookup from a loaded shared object.
                unsafe {
                    let s: libloading::Symbol<'_, *mut c_void> = lib.get(bytes)?;
                    let s = s.try_as_raw_ptr().unwrap();
                    Ok(s as *mut c_void)
                }
            }
            dbg!("starting");
            dbg!("Loading Enzyme");
            let lib = unsafe {libloading::Library::new("/home/manuel/prog/rust/build/x86_64-unknown-linux-gnu/enzyme/lib/libEnzyme-21.so")?};
            dbg!("second");
            let EnzymeSetCLBool: libloading::Symbol<'_, SetFlag> = unsafe{lib.get(b"EnzymeSetCLBool")?};
            dbg!("third");
            let registerEnzymeAndPassPipeline =
                load_ptr(&lib, b"registerEnzymeAndPassPipeline").unwrap() as *const c_void;
            dbg!("fourth");
            //let EnzymeSetCLBool: libloading::Symbol<'_, unsafe extern "C" fn(&mut c_void, u8) -> ()> = unsafe{lib.get(b"registerEnzymeAndPassPipeline")?};
            //let EnzymeSetCLBool = unsafe {EnzymeSetCLBool.try_as_raw_ptr().unwrap()};
            let EnzymeSetCLString: libloading::Symbol<'_, SetFlag> = unsafe{ lib.get(b"EnzymeSetCLString")?};
            dbg!("done");
            //let EnzymeSetCLString = unsafe {EnzymeSetCLString.try_as_raw_ptr().unwrap()};

            let EnzymePrintPerf = load_ptr(&lib, b"EnzymePrintPerf").unwrap();
            let EnzymePrintActivity = load_ptr(&lib, b"EnzymePrintActivity").unwrap();
            let EnzymePrintType = load_ptr(&lib, b"EnzymePrintType").unwrap();
            let EnzymeFunctionToAnalyze = load_ptr(&lib, b"EnzymeFunctionToAnalyze").unwrap();
            let EnzymePrint = load_ptr(&lib, b"EnzymePrint").unwrap();

            let EnzymeStrictAliasing = load_ptr(&lib, b"EnzymeStrictAliasing").unwrap();
            let looseTypeAnalysis = load_ptr(&lib, b"looseTypeAnalysis").unwrap();
            let EnzymeInline = load_ptr(&lib, b"EnzymeInline").unwrap();
            let RustTypeRules = load_ptr(&lib, b"RustTypeRules").unwrap();

            let wrap = EnzymeWrapper {
                EnzymePrintPerf,
                EnzymePrintActivity,
                EnzymePrintType,
                EnzymeFunctionToAnalyze,
                EnzymePrint,
                EnzymeStrictAliasing,
                looseTypeAnalysis,
                EnzymeInline,
                RustTypeRules,
                //EnzymeSetCLBool: EnzymeFns {set_cl: unsafe{*EnzymeSetCLBool}},
                //EnzymeSetCLString: EnzymeFns {set_cl: unsafe{*EnzymeSetCLString}},
                EnzymeSetCLBool: EnzymeFns {set_cl: *EnzymeSetCLBool},
                EnzymeSetCLString: EnzymeFns {set_cl: *EnzymeSetCLString},
                registerEnzymeAndPassPipeline,
            };
            dbg!(&wrap);
            Ok(wrap)
        }
use std::sync::Mutex;
unsafe impl Sync for EnzymeWrapper {}
unsafe impl Send for EnzymeWrapper {}
    impl EnzymeWrapper {
        pub(crate) fn current() -> &'static Mutex<EnzymeWrapper> {
            use std::sync::OnceLock;
            static CELL: OnceLock<Mutex<EnzymeWrapper>> = OnceLock::new();
            fn init_enzyme() -> Mutex<EnzymeWrapper> {
                call_dynamic().unwrap().into()
            }
            CELL.get_or_init(|| init_enzyme())
        }
        pub(crate) fn set_print_perf(&mut self, print: bool) {
            unsafe {
                //(self.EnzymeSetCLBool.set_cl)(self.EnzymePrintPerf, print as u8);
                //(self.EnzymeSetCLBool)(std::ptr::addr_of_mut!(self.EnzymePrintPerf), print as u8);
            }
        }

        pub(crate) fn set_print_activity(&mut self, print: bool) {
            unsafe {
                //(self.EnzymeSetCLBool.set_cl)(self.EnzymePrintActivity, print as u8);
                //(self.EnzymeSetCLBool)(std::ptr::addr_of_mut!(self.EnzymePrintActivity), print as u8);
            }
        }

        pub(crate) fn set_print_type(&mut self, print: bool) {
            unsafe {
               // (self.EnzymeSetCLBool.set_cl)(self.EnzymePrintType, print as u8);
            }
        }

        pub(crate) fn set_print_type_fun(&mut self, fun_name: &str) {
            let _c_fun_name = CString::new(fun_name).unwrap();
            //unsafe {
            //    (self.EnzymeSetCLString.set_cl)(
            //        self.EnzymeFunctionToAnalyze,
            //        c_fun_name.as_ptr() as *const c_char,
            //    );
            //}
        }

        pub(crate) fn set_print(&mut self, print: bool) {
            unsafe {
                //(self.EnzymeSetCLBool.set_cl)(self.EnzymePrint, print as u8);
            }
        }

        pub(crate) fn set_strict_aliasing(&mut self, strict: bool) {
            unsafe {
                //(self.EnzymeSetCLBool.set_cl)(self.EnzymeStrictAliasing, strict as u8);
            }
        }

        pub(crate) fn set_loose_types(&mut self, loose: bool) {
            unsafe {
                //(self.EnzymeSetCLBool.set_cl)(self.looseTypeAnalysis, loose as u8);
            }
        }

        pub(crate) fn set_inline(&mut self, val: bool) {
            unsafe {
                //(self.EnzymeSetCLBool.set_cl)(self.EnzymeInline, val as u8);
            }
        }

        pub(crate) fn set_rust_rules(&mut self, val: bool) {
            unsafe {
                //(self.EnzymeSetCLBool.set_cl)(self.RustTypeRules, val as u8);
            }
        }
    }


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
