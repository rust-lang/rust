//! LLVM diagnostic reports.

pub use self::OptimizationDiagnosticKind::*;
pub use self::Diagnostic::*;

use libc::c_uint;
use crate::value::Value;

use super::{DiagnosticInfo, Twine};

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

pub struct OptimizationDiagnostic<'ll> {
    pub kind: OptimizationDiagnosticKind,
    pub pass_name: String,
    pub function: &'ll Value,
    pub line: c_uint,
    pub column: c_uint,
    pub filename: String,
    pub message: String,
}

impl OptimizationDiagnostic<'ll> {
    unsafe fn unpack(
        kind: OptimizationDiagnosticKind,
        di: &'ll DiagnosticInfo,
    ) -> Self {
        let mut function = None;
        let mut line = 0;
        let mut column = 0;

        let mut message = None;
        let mut filename = None;
        let pass_name = super::build_string(|pass_name|
            message = super::build_string(|message|
                filename = super::build_string(|filename|
                    super::LLVMRustUnpackOptimizationDiagnostic(di,
                                                                pass_name,
                                                                &mut function,
                                                                &mut line,
                                                                &mut column,
                                                                filename,
                                                                message)
                ).ok()
            ).ok()
        ).ok();

        let mut filename = filename.unwrap_or_default();
        if filename.is_empty() {
            filename.push_str("<unknown file>");
        }

        OptimizationDiagnostic {
            kind,
            pass_name: pass_name.expect("got a non-UTF8 pass name from LLVM"),
            function: function.unwrap(),
            line,
            column,
            filename,
            message: message.expect("got a non-UTF8 OptimizationDiagnostic message from LLVM")
        }
    }
}

#[derive(Copy, Clone)]
pub struct InlineAsmDiagnostic<'ll> {
    pub cookie: c_uint,
    pub message: &'ll Twine,
    pub instruction: Option<&'ll Value>,
}

impl InlineAsmDiagnostic<'ll> {
    unsafe fn unpack(di: &'ll DiagnosticInfo) -> Self {
        let mut cookie = 0;
        let mut message = None;
        let mut instruction = None;

        super::LLVMRustUnpackInlineAsmDiagnostic(
            di,
            &mut cookie,
            &mut message,
            &mut instruction,
        );

        InlineAsmDiagnostic {
            cookie,
            message: message.unwrap(),
            instruction,
        }
    }
}

pub enum Diagnostic<'ll> {
    Optimization(OptimizationDiagnostic<'ll>),
    InlineAsm(InlineAsmDiagnostic<'ll>),
    PGO(&'ll DiagnosticInfo),
    Linker(&'ll DiagnosticInfo),

    /// LLVM has other types that we do not wrap here.
    UnknownDiagnostic(&'ll DiagnosticInfo),
}

impl Diagnostic<'ll> {
    pub unsafe fn unpack(di: &'ll DiagnosticInfo) -> Self {
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

            Dk::PGOProfile => {
                PGO(di)
            }
            Dk::Linker => {
                Linker(di)
            }

            _ => UnknownDiagnostic(di),
        }
    }
}
