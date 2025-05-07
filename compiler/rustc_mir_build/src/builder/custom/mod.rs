//! Provides the implementation of the `custom_mir` attribute.
//!
//! Up until MIR building, this attribute has absolutely no effect. The `mir!` macro is a normal
//! decl macro that expands like any other, and the code goes through parsing, name resolution and
//! type checking like all other code. In MIR building we finally detect whether this attribute is
//! present, and if so we branch off into this module, which implements the attribute by
//! implementing a custom lowering from THIR to MIR.
//!
//! The result of this lowering is returned "normally" from `build_mir`, with the only
//! notable difference being that the `injected` field in the body is set. Various components of the
//! MIR pipeline, like borrowck and the pass manager will then consult this field (via
//! `body.should_skip()`) to skip the parts of the MIR pipeline that precede the MIR phase the user
//! specified.
//!
//! This file defines the general framework for the custom parsing. The parsing for all the
//! "top-level" constructs can be found in the `parse` submodule, while the parsing for statements,
//! terminators, and everything below can be found in the `parse::instruction` submodule.
//!

use rustc_data_structures::fx::FxHashMap;
use rustc_hir::def_id::DefId;
use rustc_hir::{Attribute, HirId};
use rustc_index::{IndexSlice, IndexVec};
use rustc_middle::mir::*;
use rustc_middle::span_bug;
use rustc_middle::thir::*;
use rustc_middle::ty::{self, Ty, TyCtxt};
use rustc_span::Span;

mod parse;

pub(super) fn build_custom_mir<'tcx>(
    tcx: TyCtxt<'tcx>,
    did: DefId,
    hir_id: HirId,
    thir: &Thir<'tcx>,
    expr: ExprId,
    params: &IndexSlice<ParamId, Param<'tcx>>,
    return_ty: Ty<'tcx>,
    return_ty_span: Span,
    span: Span,
    attr: &Attribute,
) -> Body<'tcx> {
    let mut body = Body {
        basic_blocks: BasicBlocks::new(IndexVec::new()),
        source: MirSource::item(did),
        phase: MirPhase::Built,
        source_scopes: IndexVec::new(),
        coroutine: None,
        local_decls: IndexVec::new(),
        user_type_annotations: IndexVec::new(),
        arg_count: params.len(),
        spread_arg: None,
        var_debug_info: Vec::new(),
        span,
        required_consts: None,
        mentioned_items: None,
        is_polymorphic: false,
        tainted_by_errors: None,
        injection_phase: None,
        pass_count: 0,
        coverage_info_hi: None,
        function_coverage_info: None,
    };

    body.local_decls.push(LocalDecl::new(return_ty, return_ty_span));
    body.basic_blocks_mut().push(BasicBlockData::new(None, false));
    body.source_scopes.push(SourceScopeData {
        span,
        parent_scope: None,
        inlined: None,
        inlined_parent_scope: None,
        local_data: ClearCrossCrate::Set(SourceScopeLocalData { lint_root: hir_id }),
    });
    body.injection_phase = Some(parse_attribute(attr));

    let mut pctxt = ParseCtxt {
        tcx,
        typing_env: body.typing_env(tcx),
        thir,
        source_scope: OUTERMOST_SOURCE_SCOPE,
        body: &mut body,
        local_map: FxHashMap::default(),
        block_map: FxHashMap::default(),
    };

    let res: PResult<_> = try {
        pctxt.parse_args(params)?;
        pctxt.parse_body(expr)?;
    };
    if let Err(err) = res {
        tcx.dcx().span_fatal(
            err.span,
            format!("Could not parse {}, found: {:?}", err.expected, err.item_description),
        )
    }

    body
}

fn parse_attribute(attr: &Attribute) -> MirPhase {
    let meta_items = attr.meta_item_list().unwrap();
    let mut dialect: Option<String> = None;
    let mut phase: Option<String> = None;

    // Not handling errors properly for this internal attribute; will just abort on errors.
    for nested in meta_items {
        let name = nested.name().unwrap();
        let value = nested.value_str().unwrap().as_str().to_string();
        match name.as_str() {
            "dialect" => {
                assert!(dialect.is_none());
                dialect = Some(value);
            }
            "phase" => {
                assert!(phase.is_none());
                phase = Some(value);
            }
            other => {
                span_bug!(
                    nested.span(),
                    "Unexpected key while parsing custom_mir attribute: '{}'",
                    other
                );
            }
        }
    }

    let Some(dialect) = dialect else {
        assert!(phase.is_none());
        return MirPhase::Built;
    };

    MirPhase::parse(dialect, phase)
}

struct ParseCtxt<'a, 'tcx> {
    tcx: TyCtxt<'tcx>,
    typing_env: ty::TypingEnv<'tcx>,
    thir: &'a Thir<'tcx>,
    source_scope: SourceScope,
    body: &'a mut Body<'tcx>,
    local_map: FxHashMap<LocalVarId, Local>,
    block_map: FxHashMap<LocalVarId, BasicBlock>,
}

struct ParseError {
    span: Span,
    item_description: String,
    expected: String,
}

impl<'a, 'tcx> ParseCtxt<'a, 'tcx> {
    fn expr_error(&self, expr: ExprId, expected: &'static str) -> ParseError {
        let expr = &self.thir[expr];
        ParseError {
            span: expr.span,
            item_description: format!("{:?}", expr.kind),
            expected: expected.to_string(),
        }
    }

    fn stmt_error(&self, stmt: StmtId, expected: &'static str) -> ParseError {
        let stmt = &self.thir[stmt];
        let span = match stmt.kind {
            StmtKind::Expr { expr, .. } => self.thir[expr].span,
            StmtKind::Let { span, .. } => span,
        };
        ParseError {
            span,
            item_description: format!("{:?}", stmt.kind),
            expected: expected.to_string(),
        }
    }
}

type PResult<T> = Result<T, ParseError>;
