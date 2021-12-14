//! LLVM diagnostic reports.

pub use self::Diagnostic::*;
pub use self::OptimizationDiagnosticKind::*;

use crate::value::Value;
use libc::c_uint;

use super::{DiagnosticInfo, SMDiagnostic};
use rustc_span::InnerSpan;

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

impl<'ll> OptimizationDiagnostic<'ll> {
    unsafe fn unpack(kind: OptimizationDiagnosticKind, di: &'ll DiagnosticInfo) -> Self {
        let mut function = None;
        let mut line = 0;
        let mut column = 0;

        let mut message = None;
        let mut filename = None;
        let pass_name = super::build_string(|pass_name| {
            message = super::build_string(|message| {
                filename = super::build_string(|filename| {
                    super::LLVMRustUnpackOptimizationDiagnostic(
                        di,
                        pass_name,
                        &mut function,
                        &mut line,
                        &mut column,
                        filename,
                        message,
                    )
                })
                .ok()
            })
            .ok()
        })
        .ok();

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
            message: message.expect("got a non-UTF8 OptimizationDiagnostic message from LLVM"),
        }
    }
}

pub struct SrcMgrDiagnostic {
    pub level: super::DiagnosticLevel,
    pub message: String,
    pub source: Option<(String, Vec<InnerSpan>)>,
}

impl SrcMgrDiagnostic {
    pub unsafe fn unpack(diag: &SMDiagnostic) -> SrcMgrDiagnostic {
        // Recover the post-substitution assembly code from LLVM for better
        // diagnostics.
        let mut have_source = false;
        let mut buffer = String::new();
        let mut level = super::DiagnosticLevel::Error;
        let mut loc = 0;
        let mut ranges = [0; 8];
        let mut num_ranges = ranges.len() / 2;
        let message = super::build_string(|message| {
            buffer = super::build_string(|buffer| {
                have_source = super::LLVMRustUnpackSMDiagnostic(
                    diag,
                    message,
                    buffer,
                    &mut level,
                    &mut loc,
                    ranges.as_mut_ptr(),
                    &mut num_ranges,
                );
            })
            .expect("non-UTF8 inline asm");
        })
        .expect("non-UTF8 SMDiagnostic");

        SrcMgrDiagnostic {
            message,
            level,
            source: have_source.then(|| {
                let mut spans = vec![InnerSpan::new(loc as usize, loc as usize)];
                for i in 0..num_ranges {
                    spans.push(InnerSpan::new(ranges[i * 2] as usize, ranges[i * 2 + 1] as usize));
                }
                (buffer, spans)
            }),
        }
    }
}

#[derive(Clone)]
pub struct InlineAsmDiagnostic {
    pub level: super::DiagnosticLevel,
    pub cookie: c_uint,
    pub message: String,
    pub source: Option<(String, Vec<InnerSpan>)>,
}

impl InlineAsmDiagnostic {
    unsafe fn unpackInlineAsm(di: &DiagnosticInfo) -> Self {
        let mut cookie = 0;
        let mut message = None;
        let mut level = super::DiagnosticLevel::Error;

        super::LLVMRustUnpackInlineAsmDiagnostic(di, &mut level, &mut cookie, &mut message);

        InlineAsmDiagnostic {
            level,
            cookie,
            message: super::twine_to_string(message.unwrap()),
            source: None,
        }
    }

    unsafe fn unpackSrcMgr(di: &DiagnosticInfo) -> Self {
        let mut cookie = 0;
        let smdiag = SrcMgrDiagnostic::unpack(super::LLVMRustGetSMDiagnostic(di, &mut cookie));
        InlineAsmDiagnostic {
            level: smdiag.level,
            cookie,
            message: smdiag.message,
            source: smdiag.source,
        }
    }
}

pub enum Diagnostic<'ll> {
    Optimization(OptimizationDiagnostic<'ll>),
    InlineAsm(InlineAsmDiagnostic),
    PGO(&'ll DiagnosticInfo),
    Linker(&'ll DiagnosticInfo),
    Unsupported(&'ll DiagnosticInfo),

    /// LLVM has other types that we do not wrap here.
    UnknownDiagnostic(&'ll DiagnosticInfo),
}

impl<'ll> Diagnostic<'ll> {
    pub unsafe fn unpack(di: &'ll DiagnosticInfo) -> Self {
        use super::DiagnosticKind as Dk;
        let kind = super::LLVMRustGetDiagInfoKind(di);

        match kind {
            Dk::InlineAsm => InlineAsm(InlineAsmDiagnostic::unpackInlineAsm(di)),

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

            Dk::PGOProfile => PGO(di),
            Dk::Linker => Linker(di),
            Dk::Unsupported => Unsupported(di),

            Dk::SrcMgr => InlineAsm(InlineAsmDiagnostic::unpackSrcMgr(di)),

            _ => UnknownDiagnostic(di),
        }
    }
}
