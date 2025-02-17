//! Defines `ExpressionStore`: a lowered representation of functions, statics and
//! consts.
mod body;
mod lower;
mod pretty;
pub mod scope;

#[cfg(test)]
mod tests;

use std::ops::{Deref, Index};

use cfg::{CfgExpr, CfgOptions};
use either::Either;
use hir_expand::{name::Name, ExpandError, InFile};
use la_arena::{Arena, ArenaMap};
use rustc_hash::FxHashMap;
use smallvec::SmallVec;
use span::{Edition, MacroFileId, SyntaxContextData};
use syntax::{ast, AstPtr, SyntaxNodePtr};
use triomphe::Arc;
use tt::TextRange;

use crate::{
    db::DefDatabase,
    hir::{
        Array, AsmOperand, Binding, BindingId, Expr, ExprId, ExprOrPatId, Label, LabelId, Pat,
        PatId, RecordFieldPat, Statement,
    },
    nameres::DefMap,
    path::{ModPath, Path},
    type_ref::{TypeRef, TypeRefId, TypesMap, TypesSourceMap},
    BlockId, DefWithBodyId, Lookup, SyntheticSyntax,
};

pub use self::body::{Body, BodySourceMap};

/// A wrapper around [`span::SyntaxContextId`] that is intended only for comparisons.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct HygieneId(span::SyntaxContextId);

impl HygieneId {
    // The edition doesn't matter here, we only use this for comparisons and to lookup the macro.
    pub const ROOT: Self = Self(span::SyntaxContextId::root(Edition::Edition2015));

    pub fn new(mut ctx: span::SyntaxContextId) -> Self {
        // See `Name` for why we're doing that.
        ctx.remove_root_edition();
        Self(ctx)
    }

    pub(crate) fn lookup(self, db: &dyn DefDatabase) -> SyntaxContextData {
        db.lookup_intern_syntax_context(self.0)
    }

    pub(crate) fn is_root(self) -> bool {
        self.0.is_root()
    }
}

pub type ExprPtr = AstPtr<ast::Expr>;
pub type ExprSource = InFile<ExprPtr>;

pub type PatPtr = AstPtr<ast::Pat>;
pub type PatSource = InFile<PatPtr>;

pub type LabelPtr = AstPtr<ast::Label>;
pub type LabelSource = InFile<LabelPtr>;

pub type FieldPtr = AstPtr<ast::RecordExprField>;
pub type FieldSource = InFile<FieldPtr>;

pub type PatFieldPtr = AstPtr<Either<ast::RecordExprField, ast::RecordPatField>>;
pub type PatFieldSource = InFile<PatFieldPtr>;

pub type ExprOrPatPtr = AstPtr<Either<ast::Expr, ast::Pat>>;
pub type ExprOrPatSource = InFile<ExprOrPatPtr>;

pub type SelfParamPtr = AstPtr<ast::SelfParam>;
pub type MacroCallPtr = AstPtr<ast::MacroCall>;

#[derive(Debug, Eq, PartialEq)]
pub struct ExpressionStore {
    pub exprs: Arena<Expr>,
    pub pats: Arena<Pat>,
    pub bindings: Arena<Binding>,
    pub labels: Arena<Label>,
    /// Id of the closure/coroutine that owns the corresponding binding. If a binding is owned by the
    /// top level expression, it will not be listed in here.
    pub binding_owners: FxHashMap<BindingId, ExprId>,
    pub types: TypesMap,
    /// Block expressions in this store that may contain inner items.
    block_scopes: Box<[BlockId]>,

    /// A map from binding to its hygiene ID.
    ///
    /// Bindings that don't come from macro expansion are not allocated to save space, so not all bindings appear here.
    /// If a binding does not appear here it has `SyntaxContextId::ROOT`.
    ///
    /// Note that this may not be the direct `SyntaxContextId` of the binding's expansion, because transparent
    /// expansions are attributed to their parent expansion (recursively).
    binding_hygiene: FxHashMap<BindingId, HygieneId>,
    /// A map from an variable usages to their hygiene ID.
    ///
    /// Expressions (and destructuing patterns) that can be recorded here are single segment path, although not all single segments path refer
    /// to variables and have hygiene (some refer to items, we don't know at this stage).
    ident_hygiene: FxHashMap<ExprOrPatId, HygieneId>,
}

#[derive(Debug, Eq, PartialEq, Default)]
pub struct ExpressionStoreSourceMap {
    // AST expressions can create patterns in destructuring assignments. Therefore, `ExprSource` can also map
    // to `PatId`, and `PatId` can also map to `ExprSource` (the other way around is unaffected).
    expr_map: FxHashMap<ExprSource, ExprOrPatId>,
    expr_map_back: ArenaMap<ExprId, ExprOrPatSource>,

    pat_map: FxHashMap<PatSource, ExprOrPatId>,
    pat_map_back: ArenaMap<PatId, ExprOrPatSource>,

    label_map: FxHashMap<LabelSource, LabelId>,
    label_map_back: ArenaMap<LabelId, LabelSource>,

    binding_definitions: FxHashMap<BindingId, SmallVec<[PatId; 4]>>,

    /// We don't create explicit nodes for record fields (`S { record_field: 92 }`).
    /// Instead, we use id of expression (`92`) to identify the field.
    field_map_back: FxHashMap<ExprId, FieldSource>,
    pat_field_map_back: FxHashMap<PatId, PatFieldSource>,

    pub types: TypesSourceMap,

    template_map: Option<Box<FormatTemplate>>,

    expansions: FxHashMap<InFile<MacroCallPtr>, MacroFileId>,

    /// Diagnostics accumulated during lowering. These contain `AstPtr`s and so are stored in
    /// the source map (since they're just as volatile).
    diagnostics: Vec<ExpressionStoreDiagnostics>,
}

/// The body of an item (function, const etc.).
#[derive(Debug, Eq, PartialEq, Default)]
pub struct ExpressionStoreBuilder {
    pub exprs: Arena<Expr>,
    pub pats: Arena<Pat>,
    pub bindings: Arena<Binding>,
    pub labels: Arena<Label>,
    pub binding_owners: FxHashMap<BindingId, ExprId>,
    pub types: TypesMap,
    block_scopes: Vec<BlockId>,
    binding_hygiene: FxHashMap<BindingId, HygieneId>,
    ident_hygiene: FxHashMap<ExprOrPatId, HygieneId>,
}

#[derive(Default, Debug, Eq, PartialEq)]
struct FormatTemplate {
    /// A map from `format_args!()` expressions to their captures.
    format_args_to_captures: FxHashMap<ExprId, (HygieneId, Vec<(syntax::TextRange, Name)>)>,
    /// A map from `asm!()` expressions to their captures.
    asm_to_captures: FxHashMap<ExprId, Vec<Vec<(syntax::TextRange, usize)>>>,
    /// A map from desugared expressions of implicit captures to their source.
    ///
    /// The value stored for each capture is its template literal and offset inside it. The template literal
    /// is from the `format_args[_nl]!()` macro and so needs to be mapped up once to go to the user-written
    /// template.
    implicit_capture_to_source: FxHashMap<ExprId, InFile<(ExprPtr, TextRange)>>,
}

#[derive(Debug, Eq, PartialEq)]
pub enum ExpressionStoreDiagnostics {
    InactiveCode { node: InFile<SyntaxNodePtr>, cfg: CfgExpr, opts: CfgOptions },
    MacroError { node: InFile<MacroCallPtr>, err: ExpandError },
    UnresolvedMacroCall { node: InFile<MacroCallPtr>, path: ModPath },
    UnreachableLabel { node: InFile<AstPtr<ast::Lifetime>>, name: Name },
    AwaitOutsideOfAsync { node: InFile<AstPtr<ast::AwaitExpr>>, location: String },
    UndeclaredLabel { node: InFile<AstPtr<ast::Lifetime>>, name: Name },
}

impl ExpressionStoreBuilder {
    fn finish(self) -> ExpressionStore {
        let Self {
            block_scopes,
            mut exprs,
            mut labels,
            mut pats,
            mut bindings,
            mut binding_owners,
            mut binding_hygiene,
            mut ident_hygiene,
            mut types,
        } = self;
        exprs.shrink_to_fit();
        labels.shrink_to_fit();
        pats.shrink_to_fit();
        bindings.shrink_to_fit();
        binding_owners.shrink_to_fit();
        binding_hygiene.shrink_to_fit();
        ident_hygiene.shrink_to_fit();
        types.shrink_to_fit();

        ExpressionStore {
            exprs,
            pats,
            bindings,
            labels,
            binding_owners,
            types,
            block_scopes: block_scopes.into_boxed_slice(),
            binding_hygiene,
            ident_hygiene,
        }
    }
}

impl ExpressionStore {
    /// Returns an iterator over all block expressions in this store that define inner items.
    pub fn blocks<'a>(
        &'a self,
        db: &'a dyn DefDatabase,
    ) -> impl Iterator<Item = (BlockId, Arc<DefMap>)> + 'a {
        self.block_scopes.iter().map(move |&block| (block, db.block_def_map(block)))
    }

    pub fn walk_bindings_in_pat(&self, pat_id: PatId, mut f: impl FnMut(BindingId)) {
        self.walk_pats(pat_id, &mut |pat| {
            if let Pat::Bind { id, .. } = &self[pat] {
                f(*id);
            }
        });
    }

    pub fn walk_pats_shallow(&self, pat_id: PatId, mut f: impl FnMut(PatId)) {
        let pat = &self[pat_id];
        match pat {
            Pat::Range { .. }
            | Pat::Lit(..)
            | Pat::Path(..)
            | Pat::ConstBlock(..)
            | Pat::Wild
            | Pat::Missing
            | Pat::Expr(_) => {}
            &Pat::Bind { subpat, .. } => {
                if let Some(subpat) = subpat {
                    f(subpat);
                }
            }
            Pat::Or(args) | Pat::Tuple { args, .. } | Pat::TupleStruct { args, .. } => {
                args.iter().copied().for_each(f);
            }
            Pat::Ref { pat, .. } => f(*pat),
            Pat::Slice { prefix, slice, suffix } => {
                let total_iter = prefix.iter().chain(slice.iter()).chain(suffix.iter());
                total_iter.copied().for_each(f);
            }
            Pat::Record { args, .. } => {
                args.iter().for_each(|RecordFieldPat { pat, .. }| f(*pat));
            }
            Pat::Box { inner } => f(*inner),
        }
    }

    pub fn walk_pats(&self, pat_id: PatId, f: &mut impl FnMut(PatId)) {
        f(pat_id);
        self.walk_pats_shallow(pat_id, |p| self.walk_pats(p, f));
    }

    pub fn is_binding_upvar(&self, binding: BindingId, relative_to: ExprId) -> bool {
        match self.binding_owners.get(&binding) {
            Some(it) => {
                // We assign expression ids in a way that outer closures will receive
                // a lower id
                it.into_raw() < relative_to.into_raw()
            }
            None => true,
        }
    }

    pub fn walk_child_exprs(&self, expr_id: ExprId, mut f: impl FnMut(ExprId)) {
        let expr = &self[expr_id];
        match expr {
            Expr::Continue { .. }
            | Expr::Const(_)
            | Expr::Missing
            | Expr::Path(_)
            | Expr::OffsetOf(_)
            | Expr::Literal(_)
            | Expr::Underscore => {}
            Expr::InlineAsm(it) => it.operands.iter().for_each(|(_, op)| match op {
                AsmOperand::In { expr, .. }
                | AsmOperand::Out { expr: Some(expr), .. }
                | AsmOperand::InOut { expr, .. } => f(*expr),
                AsmOperand::SplitInOut { in_expr, out_expr, .. } => {
                    f(*in_expr);
                    if let Some(out_expr) = out_expr {
                        f(*out_expr);
                    }
                }
                AsmOperand::Out { expr: None, .. }
                | AsmOperand::Const(_)
                | AsmOperand::Label(_)
                | AsmOperand::Sym(_) => (),
            }),
            Expr::If { condition, then_branch, else_branch } => {
                f(*condition);
                f(*then_branch);
                if let &Some(else_branch) = else_branch {
                    f(else_branch);
                }
            }
            Expr::Let { expr, pat } => {
                self.walk_exprs_in_pat(*pat, &mut f);
                f(*expr);
            }
            Expr::Block { statements, tail, .. }
            | Expr::Unsafe { statements, tail, .. }
            | Expr::Async { statements, tail, .. } => {
                for stmt in statements.iter() {
                    match stmt {
                        Statement::Let { initializer, else_branch, pat, .. } => {
                            if let &Some(expr) = initializer {
                                f(expr);
                            }
                            if let &Some(expr) = else_branch {
                                f(expr);
                            }
                            self.walk_exprs_in_pat(*pat, &mut f);
                        }
                        Statement::Expr { expr: expression, .. } => f(*expression),
                        Statement::Item(_) => (),
                    }
                }
                if let &Some(expr) = tail {
                    f(expr);
                }
            }
            Expr::Loop { body, .. } => f(*body),
            Expr::Call { callee, args, .. } => {
                f(*callee);
                args.iter().copied().for_each(f);
            }
            Expr::MethodCall { receiver, args, .. } => {
                f(*receiver);
                args.iter().copied().for_each(f);
            }
            Expr::Match { expr, arms } => {
                f(*expr);
                arms.iter().for_each(|arm| {
                    f(arm.expr);
                    self.walk_exprs_in_pat(arm.pat, &mut f);
                });
            }
            Expr::Break { expr, .. }
            | Expr::Return { expr }
            | Expr::Yield { expr }
            | Expr::Yeet { expr } => {
                if let &Some(expr) = expr {
                    f(expr);
                }
            }
            Expr::Become { expr } => f(*expr),
            Expr::RecordLit { fields, spread, .. } => {
                for field in fields.iter() {
                    f(field.expr);
                }
                if let &Some(expr) = spread {
                    f(expr);
                }
            }
            Expr::Closure { body, .. } => {
                f(*body);
            }
            Expr::BinaryOp { lhs, rhs, .. } => {
                f(*lhs);
                f(*rhs);
            }
            Expr::Range { lhs, rhs, .. } => {
                if let &Some(lhs) = rhs {
                    f(lhs);
                }
                if let &Some(rhs) = lhs {
                    f(rhs);
                }
            }
            Expr::Index { base, index, .. } => {
                f(*base);
                f(*index);
            }
            Expr::Field { expr, .. }
            | Expr::Await { expr }
            | Expr::Cast { expr, .. }
            | Expr::Ref { expr, .. }
            | Expr::UnaryOp { expr, .. }
            | Expr::Box { expr } => {
                f(*expr);
            }
            Expr::Tuple { exprs, .. } => exprs.iter().copied().for_each(f),
            Expr::Array(a) => match a {
                Array::ElementList { elements, .. } => elements.iter().copied().for_each(f),
                Array::Repeat { initializer, repeat } => {
                    f(*initializer);
                    f(*repeat)
                }
            },
            &Expr::Assignment { target, value } => {
                self.walk_exprs_in_pat(target, &mut f);
                f(value);
            }
        }
    }

    pub fn walk_child_exprs_without_pats(&self, expr_id: ExprId, mut f: impl FnMut(ExprId)) {
        let expr = &self[expr_id];
        match expr {
            Expr::Continue { .. }
            | Expr::Const(_)
            | Expr::Missing
            | Expr::Path(_)
            | Expr::OffsetOf(_)
            | Expr::Literal(_)
            | Expr::Underscore => {}
            Expr::InlineAsm(it) => it.operands.iter().for_each(|(_, op)| match op {
                AsmOperand::In { expr, .. }
                | AsmOperand::Out { expr: Some(expr), .. }
                | AsmOperand::InOut { expr, .. } => f(*expr),
                AsmOperand::SplitInOut { in_expr, out_expr, .. } => {
                    f(*in_expr);
                    if let Some(out_expr) = out_expr {
                        f(*out_expr);
                    }
                }
                AsmOperand::Out { expr: None, .. }
                | AsmOperand::Const(_)
                | AsmOperand::Label(_)
                | AsmOperand::Sym(_) => (),
            }),
            Expr::If { condition, then_branch, else_branch } => {
                f(*condition);
                f(*then_branch);
                if let &Some(else_branch) = else_branch {
                    f(else_branch);
                }
            }
            Expr::Let { expr, .. } => {
                f(*expr);
            }
            Expr::Block { statements, tail, .. }
            | Expr::Unsafe { statements, tail, .. }
            | Expr::Async { statements, tail, .. } => {
                for stmt in statements.iter() {
                    match stmt {
                        Statement::Let { initializer, else_branch, .. } => {
                            if let &Some(expr) = initializer {
                                f(expr);
                            }
                            if let &Some(expr) = else_branch {
                                f(expr);
                            }
                        }
                        Statement::Expr { expr: expression, .. } => f(*expression),
                        Statement::Item(_) => (),
                    }
                }
                if let &Some(expr) = tail {
                    f(expr);
                }
            }
            Expr::Loop { body, .. } => f(*body),
            Expr::Call { callee, args, .. } => {
                f(*callee);
                args.iter().copied().for_each(f);
            }
            Expr::MethodCall { receiver, args, .. } => {
                f(*receiver);
                args.iter().copied().for_each(f);
            }
            Expr::Match { expr, arms } => {
                f(*expr);
                arms.iter().map(|arm| arm.expr).for_each(f);
            }
            Expr::Break { expr, .. }
            | Expr::Return { expr }
            | Expr::Yield { expr }
            | Expr::Yeet { expr } => {
                if let &Some(expr) = expr {
                    f(expr);
                }
            }
            Expr::Become { expr } => f(*expr),
            Expr::RecordLit { fields, spread, .. } => {
                for field in fields.iter() {
                    f(field.expr);
                }
                if let &Some(expr) = spread {
                    f(expr);
                }
            }
            Expr::Closure { body, .. } => {
                f(*body);
            }
            Expr::BinaryOp { lhs, rhs, .. } => {
                f(*lhs);
                f(*rhs);
            }
            Expr::Range { lhs, rhs, .. } => {
                if let &Some(lhs) = rhs {
                    f(lhs);
                }
                if let &Some(rhs) = lhs {
                    f(rhs);
                }
            }
            Expr::Index { base, index, .. } => {
                f(*base);
                f(*index);
            }
            Expr::Field { expr, .. }
            | Expr::Await { expr }
            | Expr::Cast { expr, .. }
            | Expr::Ref { expr, .. }
            | Expr::UnaryOp { expr, .. }
            | Expr::Box { expr } => {
                f(*expr);
            }
            Expr::Tuple { exprs, .. } => exprs.iter().copied().for_each(f),
            Expr::Array(a) => match a {
                Array::ElementList { elements, .. } => elements.iter().copied().for_each(f),
                Array::Repeat { initializer, repeat } => {
                    f(*initializer);
                    f(*repeat)
                }
            },
            &Expr::Assignment { target: _, value } => f(value),
        }
    }

    pub fn walk_exprs_in_pat(&self, pat_id: PatId, f: &mut impl FnMut(ExprId)) {
        self.walk_pats(pat_id, &mut |pat| {
            if let Pat::Expr(expr) | Pat::ConstBlock(expr) = self[pat] {
                f(expr);
            }
        });
    }

    fn binding_hygiene(&self, binding: BindingId) -> HygieneId {
        self.binding_hygiene.get(&binding).copied().unwrap_or(HygieneId::ROOT)
    }

    pub fn expr_path_hygiene(&self, expr: ExprId) -> HygieneId {
        self.ident_hygiene.get(&expr.into()).copied().unwrap_or(HygieneId::ROOT)
    }

    pub fn pat_path_hygiene(&self, pat: PatId) -> HygieneId {
        self.ident_hygiene.get(&pat.into()).copied().unwrap_or(HygieneId::ROOT)
    }

    pub fn expr_or_pat_path_hygiene(&self, id: ExprOrPatId) -> HygieneId {
        match id {
            ExprOrPatId::ExprId(id) => self.expr_path_hygiene(id),
            ExprOrPatId::PatId(id) => self.pat_path_hygiene(id),
        }
    }
}

impl Index<ExprId> for ExpressionStore {
    type Output = Expr;

    fn index(&self, expr: ExprId) -> &Expr {
        &self.exprs[expr]
    }
}

impl Index<PatId> for ExpressionStore {
    type Output = Pat;

    fn index(&self, pat: PatId) -> &Pat {
        &self.pats[pat]
    }
}

impl Index<LabelId> for ExpressionStore {
    type Output = Label;

    fn index(&self, label: LabelId) -> &Label {
        &self.labels[label]
    }
}

impl Index<BindingId> for ExpressionStore {
    type Output = Binding;

    fn index(&self, b: BindingId) -> &Binding {
        &self.bindings[b]
    }
}

impl Index<TypeRefId> for ExpressionStore {
    type Output = TypeRef;

    fn index(&self, b: TypeRefId) -> &TypeRef {
        &self.types[b]
    }
}

// FIXME: Change `node_` prefix to something more reasonable.
// Perhaps `expr_syntax` and `expr_id`?
impl ExpressionStoreSourceMap {
    pub fn expr_or_pat_syntax(&self, id: ExprOrPatId) -> Result<ExprOrPatSource, SyntheticSyntax> {
        match id {
            ExprOrPatId::ExprId(id) => self.expr_syntax(id),
            ExprOrPatId::PatId(id) => self.pat_syntax(id),
        }
    }

    pub fn expr_syntax(&self, expr: ExprId) -> Result<ExprOrPatSource, SyntheticSyntax> {
        self.expr_map_back.get(expr).cloned().ok_or(SyntheticSyntax)
    }

    pub fn node_expr(&self, node: InFile<&ast::Expr>) -> Option<ExprOrPatId> {
        let src = node.map(AstPtr::new);
        self.expr_map.get(&src).cloned()
    }

    pub fn node_macro_file(&self, node: InFile<&ast::MacroCall>) -> Option<MacroFileId> {
        let src = node.map(AstPtr::new);
        self.expansions.get(&src).cloned()
    }

    pub fn macro_calls(&self) -> impl Iterator<Item = (InFile<MacroCallPtr>, MacroFileId)> + '_ {
        self.expansions.iter().map(|(&a, &b)| (a, b))
    }

    pub fn pat_syntax(&self, pat: PatId) -> Result<ExprOrPatSource, SyntheticSyntax> {
        self.pat_map_back.get(pat).cloned().ok_or(SyntheticSyntax)
    }

    pub fn node_pat(&self, node: InFile<&ast::Pat>) -> Option<ExprOrPatId> {
        self.pat_map.get(&node.map(AstPtr::new)).cloned()
    }

    pub fn label_syntax(&self, label: LabelId) -> LabelSource {
        self.label_map_back[label]
    }

    pub fn patterns_for_binding(&self, binding: BindingId) -> &[PatId] {
        self.binding_definitions.get(&binding).map_or(&[], Deref::deref)
    }

    pub fn node_label(&self, node: InFile<&ast::Label>) -> Option<LabelId> {
        let src = node.map(AstPtr::new);
        self.label_map.get(&src).cloned()
    }

    pub fn field_syntax(&self, expr: ExprId) -> FieldSource {
        self.field_map_back[&expr]
    }

    pub fn pat_field_syntax(&self, pat: PatId) -> PatFieldSource {
        self.pat_field_map_back[&pat]
    }

    pub fn macro_expansion_expr(&self, node: InFile<&ast::MacroExpr>) -> Option<ExprOrPatId> {
        let src = node.map(AstPtr::new).map(AstPtr::upcast::<ast::MacroExpr>).map(AstPtr::upcast);
        self.expr_map.get(&src).copied()
    }

    pub fn expansions(&self) -> impl Iterator<Item = (&InFile<MacroCallPtr>, &MacroFileId)> {
        self.expansions.iter()
    }

    pub fn implicit_format_args(
        &self,
        node: InFile<&ast::FormatArgsExpr>,
    ) -> Option<(HygieneId, &[(syntax::TextRange, Name)])> {
        let src = node.map(AstPtr::new).map(AstPtr::upcast::<ast::Expr>);
        let (hygiene, names) = self
            .template_map
            .as_ref()?
            .format_args_to_captures
            .get(&self.expr_map.get(&src)?.as_expr()?)?;
        Some((*hygiene, &**names))
    }

    pub fn format_args_implicit_capture(
        &self,
        capture_expr: ExprId,
    ) -> Option<InFile<(ExprPtr, TextRange)>> {
        self.template_map.as_ref()?.implicit_capture_to_source.get(&capture_expr).copied()
    }

    pub fn asm_template_args(
        &self,
        node: InFile<&ast::AsmExpr>,
    ) -> Option<(ExprId, &[Vec<(syntax::TextRange, usize)>])> {
        let src = node.map(AstPtr::new).map(AstPtr::upcast::<ast::Expr>);
        let expr = self.expr_map.get(&src)?.as_expr()?;
        Some(expr)
            .zip(self.template_map.as_ref()?.asm_to_captures.get(&expr).map(std::ops::Deref::deref))
    }

    /// Get a reference to the source map's diagnostics.
    pub fn diagnostics(&self) -> &[ExpressionStoreDiagnostics] {
        &self.diagnostics
    }

    fn shrink_to_fit(&mut self) {
        let Self {
            expr_map,
            expr_map_back,
            pat_map,
            pat_map_back,
            label_map,
            label_map_back,
            field_map_back,
            pat_field_map_back,
            expansions,
            template_map,
            diagnostics,
            binding_definitions,
            types,
        } = self;
        if let Some(template_map) = template_map {
            let FormatTemplate {
                format_args_to_captures,
                asm_to_captures,
                implicit_capture_to_source,
            } = &mut **template_map;
            format_args_to_captures.shrink_to_fit();
            asm_to_captures.shrink_to_fit();
            implicit_capture_to_source.shrink_to_fit();
        }
        expr_map.shrink_to_fit();
        expr_map_back.shrink_to_fit();
        pat_map.shrink_to_fit();
        pat_map_back.shrink_to_fit();
        label_map.shrink_to_fit();
        label_map_back.shrink_to_fit();
        field_map_back.shrink_to_fit();
        pat_field_map_back.shrink_to_fit();
        expansions.shrink_to_fit();
        diagnostics.shrink_to_fit();
        binding_definitions.shrink_to_fit();
        types.shrink_to_fit();
    }
}
