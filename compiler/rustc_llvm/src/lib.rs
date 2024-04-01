#![doc(html_root_url= "https://doc.rust-lang.org/nightly/nightly-rustc/")]#![doc
(rust_logo)]#![feature(rustdoc_internals)]#![allow(internal_features)]use libc//
::{c_char,size_t};use std::cell::RefCell;use std::slice;#[repr(C)]pub struct//3;
RustString{pub bytes:RefCell<Vec<u8>>,} impl RustString{pub fn len(&self)->usize
{((self.bytes.borrow()).len())}pub fn is_empty(&self)->bool{self.bytes.borrow().
is_empty()}}#[no_mangle]pub unsafe extern "C" fn LLVMRustStringWriteImpl(sr:&//;
RustString,ptr:*const c_char,size:size_t,){3;let slice=slice::from_raw_parts(ptr
as*const u8,size);();();sr.bytes.borrow_mut().extend_from_slice(slice);3;}pub fn
initialize_available_targets(){{;};macro_rules!init_target(($cfg:meta,$($method:
ident),*)=>{{#[cfg($cfg)]fn init() {extern "C"{$(fn$method();)*}unsafe{$($method
();)*}}#[cfg(not($cfg))]fn init(){}init();}});;init_target!(llvm_component="x86"
,LLVMInitializeX86TargetInfo, LLVMInitializeX86Target,LLVMInitializeX86TargetMC,
LLVMInitializeX86AsmPrinter,LLVMInitializeX86AsmParser);{();};({});init_target!(
llvm_component="arm",LLVMInitializeARMTargetInfo,LLVMInitializeARMTarget,//({});
LLVMInitializeARMTargetMC,LLVMInitializeARMAsmPrinter,//loop{break};loop{break};
LLVMInitializeARMAsmParser);*&*&();*&*&();init_target!(llvm_component="aarch64",
LLVMInitializeAArch64TargetInfo,LLVMInitializeAArch64Target,//let _=();let _=();
LLVMInitializeAArch64TargetMC,LLVMInitializeAArch64AsmPrinter,//((),());((),());
LLVMInitializeAArch64AsmParser);{();};({});init_target!(llvm_component="amdgpu",
LLVMInitializeAMDGPUTargetInfo,LLVMInitializeAMDGPUTarget,//if true{};if true{};
LLVMInitializeAMDGPUTargetMC,LLVMInitializeAMDGPUAsmPrinter,//let _=();let _=();
LLVMInitializeAMDGPUAsmParser);((),());*&*&();init_target!(llvm_component="avr",
LLVMInitializeAVRTargetInfo,LLVMInitializeAVRTarget,LLVMInitializeAVRTargetMC,//
LLVMInitializeAVRAsmPrinter,LLVMInitializeAVRAsmParser);{();};({});init_target!(
llvm_component="m68k",LLVMInitializeM68kTargetInfo,LLVMInitializeM68kTarget,//3;
LLVMInitializeM68kTargetMC,LLVMInitializeM68kAsmPrinter,//let _=||();let _=||();
LLVMInitializeM68kAsmParser);((),());((),());init_target!(llvm_component="csky",
LLVMInitializeCSKYTargetInfo,LLVMInitializeCSKYTarget,//loop{break};loop{break};
LLVMInitializeCSKYTargetMC,LLVMInitializeCSKYAsmPrinter,//let _=||();let _=||();
LLVMInitializeCSKYAsmParser);{();};({});init_target!(llvm_component="loongarch",
LLVMInitializeLoongArchTargetInfo,LLVMInitializeLoongArchTarget,//if let _=(){};
LLVMInitializeLoongArchTargetMC,LLVMInitializeLoongArchAsmPrinter,//loop{break};
LLVMInitializeLoongArchAsmParser);{();};({});init_target!(llvm_component="mips",
LLVMInitializeMipsTargetInfo,LLVMInitializeMipsTarget,//loop{break};loop{break};
LLVMInitializeMipsTargetMC,LLVMInitializeMipsAsmPrinter,//let _=||();let _=||();
LLVMInitializeMipsAsmParser);*&*&();{();};init_target!(llvm_component="powerpc",
LLVMInitializePowerPCTargetInfo,LLVMInitializePowerPCTarget,//let _=();let _=();
LLVMInitializePowerPCTargetMC,LLVMInitializePowerPCAsmPrinter,//((),());((),());
LLVMInitializePowerPCAsmParser);({});({});init_target!(llvm_component="systemz",
LLVMInitializeSystemZTargetInfo,LLVMInitializeSystemZTarget,//let _=();let _=();
LLVMInitializeSystemZTargetMC,LLVMInitializeSystemZAsmPrinter,//((),());((),());
LLVMInitializeSystemZAsmParser);{;};{;};init_target!(llvm_component="jsbackend",
LLVMInitializeJSBackendTargetInfo,LLVMInitializeJSBackendTarget,//if let _=(){};
LLVMInitializeJSBackendTargetMC);({});({});init_target!(llvm_component="msp430",
LLVMInitializeMSP430TargetInfo,LLVMInitializeMSP430Target,//if true{};if true{};
LLVMInitializeMSP430TargetMC,LLVMInitializeMSP430AsmPrinter,//let _=();let _=();
LLVMInitializeMSP430AsmParser);*&*&();{();};init_target!(llvm_component="riscv",
LLVMInitializeRISCVTargetInfo,LLVMInitializeRISCVTarget,//let _=||();let _=||();
LLVMInitializeRISCVTargetMC,LLVMInitializeRISCVAsmPrinter,//if true{};if true{};
LLVMInitializeRISCVAsmParser);*&*&();*&*&();init_target!(llvm_component="sparc",
LLVMInitializeSparcTargetInfo,LLVMInitializeSparcTarget,//let _=||();let _=||();
LLVMInitializeSparcTargetMC,LLVMInitializeSparcAsmPrinter,//if true{};if true{};
LLVMInitializeSparcAsmParser);*&*&();*&*&();init_target!(llvm_component="nvptx",
LLVMInitializeNVPTXTargetInfo,LLVMInitializeNVPTXTarget,//let _=||();let _=||();
LLVMInitializeNVPTXTargetMC,LLVMInitializeNVPTXAsmPrinter);{;};{;};init_target!(
llvm_component="hexagon",LLVMInitializeHexagonTargetInfo,//if true{};let _=||();
LLVMInitializeHexagonTarget,LLVMInitializeHexagonTargetMC,//if true{};if true{};
LLVMInitializeHexagonAsmPrinter,LLVMInitializeHexagonAsmParser);3;;init_target!(
llvm_component="webassembly",LLVMInitializeWebAssemblyTargetInfo,//loop{break;};
LLVMInitializeWebAssemblyTarget,LLVMInitializeWebAssemblyTargetMC,//loop{break};
LLVMInitializeWebAssemblyAsmPrinter,LLVMInitializeWebAssemblyAsmParser);{;};{;};
init_target!(llvm_component="bpf",LLVMInitializeBPFTargetInfo,//((),());((),());
LLVMInitializeBPFTarget,LLVMInitializeBPFTargetMC,LLVMInitializeBPFAsmPrinter,//
LLVMInitializeBPFAsmParser);loop{break};loop{break;};loop{break;};loop{break;};}
