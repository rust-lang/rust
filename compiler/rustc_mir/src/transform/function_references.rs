use rustc_hir::def_id::DefId;
use rustc_middle::mir::visit::Visitor;
use rustc_middle::mir::*;
use rustc_middle::ty::{self, Ty, TyCtxt};
use rustc_session::lint::builtin::FUNCTION_REFERENCES;
use rustc_span::Span;
use rustc_target::spec::abi::Abi;

use crate::transform::{MirPass, MirSource};

pub struct FunctionReferences;

impl<'tcx> MirPass<'tcx> for FunctionReferences {
    fn run_pass(&self, tcx: TyCtxt<'tcx>, _src: MirSource<'tcx>, body: &mut Body<'tcx>) {
        let source_info = SourceInfo::outermost(body.span);
        let mut checker = FunctionRefChecker {
            tcx,
            body,
            potential_lints: Vec::new(),
            casts: Vec::new(),
            calls: Vec::new(),
            source_info,
        };
        checker.visit_body(&body);
    }
}

struct FunctionRefChecker<'a, 'tcx> {
    tcx: TyCtxt<'tcx>,
    body: &'a Body<'tcx>,
    potential_lints: Vec<FunctionRefLint>,
    casts: Vec<Span>,
    calls: Vec<Span>,
    source_info: SourceInfo,
}

impl<'a, 'tcx> Visitor<'tcx> for FunctionRefChecker<'a, 'tcx> {
    fn visit_basic_block_data(&mut self, block: BasicBlock, data: &BasicBlockData<'tcx>) {
        self.super_basic_block_data(block, data);
        for cast_span in self.casts.drain(..) {
            self.potential_lints.retain(|lint| lint.source_info.span != cast_span);
        }
        for call_span in self.calls.drain(..) {
            self.potential_lints.retain(|lint| lint.source_info.span != call_span);
        }
        for lint in self.potential_lints.drain(..) {
            lint.emit(self.tcx, self.body);
        }
    }
    fn visit_statement(&mut self, statement: &Statement<'tcx>, location: Location) {
        self.source_info = statement.source_info;
        self.super_statement(statement, location);
    }
    fn visit_terminator(&mut self, terminator: &Terminator<'tcx>, location: Location) {
        self.source_info = terminator.source_info;
        if let TerminatorKind::Call {
            func,
            args: _,
            destination: _,
            cleanup: _,
            from_hir_call: _,
            fn_span: _,
        } = &terminator.kind
        {
            let span = match func {
                Operand::Copy(place) | Operand::Move(place) => {
                    self.body.local_decls[place.local].source_info.span
                }
                Operand::Constant(constant) => constant.span,
            };
            self.calls.push(span);
        };
        self.super_terminator(terminator, location);
    }
    fn visit_rvalue(&mut self, rvalue: &Rvalue<'tcx>, location: Location) {
        match rvalue {
            Rvalue::Ref(_, _, place) | Rvalue::AddressOf(_, place) => {
                let decl = &self.body.local_decls[place.local];
                if let ty::FnDef(def_id, _) = decl.ty.kind {
                    let ident = self
                        .body
                        .var_debug_info
                        .iter()
                        .find(|info| info.source_info.span == decl.source_info.span)
                        .map(|info| info.name.to_ident_string())
                        .unwrap_or(self.tcx.def_path_str(def_id));
                    let lint = FunctionRefLint { ident, def_id, source_info: self.source_info };
                    self.potential_lints.push(lint);
                }
            }
            Rvalue::Cast(_, op, _) => {
                let op_ty = op.ty(self.body, self.tcx);
                if self.is_fn_ref(op_ty) {
                    self.casts.push(self.source_info.span);
                }
            }
            _ => {}
        }
        self.super_rvalue(rvalue, location);
    }
}

impl<'a, 'tcx> FunctionRefChecker<'a, 'tcx> {
    fn is_fn_ref(&self, ty: Ty<'tcx>) -> bool {
        let referent_ty = match ty.kind {
            ty::Ref(_, referent_ty, _) => Some(referent_ty),
            ty::RawPtr(ty_and_mut) => Some(ty_and_mut.ty),
            _ => None,
        };
        referent_ty
            .map(|ref_ty| if let ty::FnDef(..) = ref_ty.kind { true } else { false })
            .unwrap_or(false)
    }
}

struct FunctionRefLint {
    ident: String,
    def_id: DefId,
    source_info: SourceInfo,
}

impl<'tcx> FunctionRefLint {
    fn emit(&self, tcx: TyCtxt<'tcx>, body: &Body<'tcx>) {
        let def_id = self.def_id;
        let source_info = self.source_info;
        let lint_root = body.source_scopes[source_info.scope]
            .local_data
            .as_ref()
            .assert_crate_local()
            .lint_root;
        let fn_sig = tcx.fn_sig(def_id);
        let unsafety = fn_sig.unsafety().prefix_str();
        let abi = match fn_sig.abi() {
            Abi::Rust => String::from(""),
            other_abi => {
                let mut s = String::from("extern \"");
                s.push_str(other_abi.name());
                s.push_str("\" ");
                s
            }
        };
        let num_args = fn_sig.inputs().map_bound(|inputs| inputs.len()).skip_binder();
        let variadic = if fn_sig.c_variadic() { ", ..." } else { "" };
        let ret = if fn_sig.output().skip_binder().is_unit() { "" } else { " -> _" };
        tcx.struct_span_lint_hir(FUNCTION_REFERENCES, lint_root, source_info.span, |lint| {
            lint.build(&format!(
                "cast `{}` with `as {}{}fn({}{}){}` to use it as a pointer",
                self.ident,
                unsafety,
                abi,
                vec!["_"; num_args].join(", "),
                variadic,
                ret,
            ))
            .emit()
        });
    }
}
