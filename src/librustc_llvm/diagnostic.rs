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

use libc::{c_char, c_uint};
use std::ptr;

use {ValueRef, TwineRef, DebugLocRef, DiagnosticInfoRef};

#[derive(Copy, Clone)]
pub enum OptimizationDiagnosticKind {
    OptimizationRemark,
    OptimizationMissed,
    OptimizationAnalysis,
    OptimizationFailure,
}

impl OptimizationDiagnosticKind {
    pub fn describe(self) -> &'static str {
        match self {
            OptimizationRemark => "remark",
            OptimizationMissed => "missed",
            OptimizationAnalysis => "analysis",
            OptimizationFailure => "failure",
        }
    }
}

#[derive(Copy, Clone)]
pub struct OptimizationDiagnostic {
    pub kind: OptimizationDiagnosticKind,
    pub pass_name: *const c_char,
    pub function: ValueRef,
    pub debug_loc: DebugLocRef,
    pub message: TwineRef,
}

impl OptimizationDiagnostic {
    unsafe fn unpack(kind: OptimizationDiagnosticKind, di: DiagnosticInfoRef)
            -> OptimizationDiagnostic {

        let mut opt = OptimizationDiagnostic {
            kind: kind,
            pass_name: ptr::null(),
            function: ptr::null_mut(),
            debug_loc: ptr::null_mut(),
            message: ptr::null_mut(),
        };

        super::LLVMUnpackOptimizationDiagnostic(di,
            &mut opt.pass_name,
            &mut opt.function,
            &mut opt.debug_loc,
            &mut opt.message);

        opt
    }
}

#[derive(Copy, Clone)]
pub struct InlineAsmDiagnostic {
    pub cookie: c_uint,
    pub message: TwineRef,
    pub instruction: ValueRef,
}

impl InlineAsmDiagnostic {
    unsafe fn unpack(di: DiagnosticInfoRef)
            -> InlineAsmDiagnostic {

        let mut opt = InlineAsmDiagnostic {
            cookie: 0,
            message: ptr::null_mut(),
            instruction: ptr::null_mut(),
        };

        super::LLVMUnpackInlineAsmDiagnostic(di,
            &mut opt.cookie,
            &mut opt.message,
            &mut opt.instruction);

        opt
    }
}

#[derive(Copy, Clone)]
pub enum Diagnostic {
    Optimization(OptimizationDiagnostic),
    InlineAsm(InlineAsmDiagnostic),

    /// LLVM has other types that we do not wrap here.
    UnknownDiagnostic(DiagnosticInfoRef),
}

impl Diagnostic {
    pub unsafe fn unpack(di: DiagnosticInfoRef) -> Diagnostic {
        let kind = super::LLVMGetDiagInfoKind(di);

        match kind {
            super::DK_InlineAsm
                => InlineAsm(InlineAsmDiagnostic::unpack(di)),

            super::DK_OptimizationRemark
                => Optimization(OptimizationDiagnostic::unpack(OptimizationRemark, di)),

            super::DK_OptimizationRemarkMissed
                => Optimization(OptimizationDiagnostic::unpack(OptimizationMissed, di)),

            super::DK_OptimizationRemarkAnalysis
                => Optimization(OptimizationDiagnostic::unpack(OptimizationAnalysis, di)),

            super::DK_OptimizationFailure
                => Optimization(OptimizationDiagnostic::unpack(OptimizationFailure, di)),

            _ => UnknownDiagnostic(di)
        }
    }
}
