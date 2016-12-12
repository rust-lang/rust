// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! LLVM diagnostic reports.

pub use self::OptimizationDiagnosticKind::*;
pub use self::Diagnostic::*;

use libc::c_uint;
use std::ptr;

use {DiagnosticInfoRef, TwineRef, ValueRef};
use ffi::DebugLocRef;

#[derive(Copy, Clone)]
pub enum OptimizationDiagnosticKind {
    OptimizationRemark,
    OptimizationMissed,
    OptimizationAnalysis,
    OptimizationAnalysisFPCommute,
    OptimizationAnalysisAliasing,
    OptimizationFailure,
    OptimizationRemarkOther,
}

impl OptimizationDiagnosticKind {
    pub fn describe(self) -> &'static str {
        match self {
            OptimizationRemark | OptimizationRemarkOther => "remark",
            OptimizationMissed => "missed",
            OptimizationAnalysis => "analysis",
            OptimizationAnalysisFPCommute => "floating-point",
            OptimizationAnalysisAliasing => "aliasing",
            OptimizationFailure => "failure",
        }
    }
}

pub struct OptimizationDiagnostic {
    pub kind: OptimizationDiagnosticKind,
    pub pass_name: String,
    pub function: ValueRef,
    pub debug_loc: DebugLocRef,
    pub message: String,
}

impl OptimizationDiagnostic {
    unsafe fn unpack(kind: OptimizationDiagnosticKind,
                     di: DiagnosticInfoRef)
                     -> OptimizationDiagnostic {
        let mut function = ptr::null_mut();
        let mut debug_loc = ptr::null_mut();

        let mut message = None;
        let pass_name = super::build_string(|pass_name|
            message = super::build_string(|message|
                super::LLVMRustUnpackOptimizationDiagnostic(di,
                                                            pass_name,
                                                            &mut function,
                                                            &mut debug_loc,
                                                            message)
            )
        );

        OptimizationDiagnostic {
            kind: kind,
            pass_name: pass_name.expect("got a non-UTF8 pass name from LLVM"),
            function: function,
            debug_loc: debug_loc,
            message: message.expect("got a non-UTF8 OptimizationDiagnostic message from LLVM")
        }
    }
}

#[derive(Copy, Clone)]
pub struct InlineAsmDiagnostic {
    pub cookie: c_uint,
    pub message: TwineRef,
    pub instruction: ValueRef,
}

impl InlineAsmDiagnostic {
    unsafe fn unpack(di: DiagnosticInfoRef) -> InlineAsmDiagnostic {

        let mut opt = InlineAsmDiagnostic {
            cookie: 0,
            message: ptr::null_mut(),
            instruction: ptr::null_mut(),
        };

        super::LLVMRustUnpackInlineAsmDiagnostic(di,
                                                 &mut opt.cookie,
                                                 &mut opt.message,
                                                 &mut opt.instruction);

        opt
    }
}

pub enum Diagnostic {
    Optimization(OptimizationDiagnostic),
    InlineAsm(InlineAsmDiagnostic),

    /// LLVM has other types that we do not wrap here.
    UnknownDiagnostic(DiagnosticInfoRef),
}

impl Diagnostic {
    pub unsafe fn unpack(di: DiagnosticInfoRef) -> Diagnostic {
        use super::DiagnosticKind as Dk;
        let kind = super::LLVMRustGetDiagInfoKind(di);

        match kind {
            Dk::InlineAsm => InlineAsm(InlineAsmDiagnostic::unpack(di)),

            Dk::OptimizationRemark => {
                Optimization(OptimizationDiagnostic::unpack(OptimizationRemark, di))
            }
            Dk::OptimizationRemarkOther => {
                Optimization(OptimizationDiagnostic::unpack(OptimizationRemarkOther, di))
            }
            Dk::OptimizationRemarkMissed => {
                Optimization(OptimizationDiagnostic::unpack(OptimizationMissed, di))
            }

            Dk::OptimizationRemarkAnalysis => {
                Optimization(OptimizationDiagnostic::unpack(OptimizationAnalysis, di))
            }

            Dk::OptimizationRemarkAnalysisFPCommute => {
                Optimization(OptimizationDiagnostic::unpack(OptimizationAnalysisFPCommute, di))
            }

            Dk::OptimizationRemarkAnalysisAliasing => {
                Optimization(OptimizationDiagnostic::unpack(OptimizationAnalysisAliasing, di))
            }

            Dk::OptimizationFailure => {
                Optimization(OptimizationDiagnostic::unpack(OptimizationFailure, di))
            }

            _ => UnknownDiagnostic(di),
        }
    }
}
