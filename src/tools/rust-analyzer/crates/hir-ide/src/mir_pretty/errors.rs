//! Pretty-printing MIR and consteval errors.

use std::fmt::Write;

use either::Either;
use hir_def::{
    GenericParamId, ItemContainerId, Lookup,
    expr_store::{Body, ExpressionStore},
    hir::generics::GenericParams,
    signatures::{FunctionSignature, TraitSignature},
};
use hir_expand::{InFile, name::Name};
use hir_ty::{
    InferBodyId,
    consteval::ConstEvalError,
    db::HirDatabase,
    mir::{MirEvalError, MirLowerError, MirSpan},
};
use macros::extension;
use span::{FileId, TextRange};
use syntax::SyntaxNodePtr;

use crate::display::{ClosureStyle, DisplayTarget, HirDisplay};

#[extension(pub trait MirLowerErrorPretty)]
impl MirLowerError<'_> {
    fn pretty_print(
        &self,
        f: &mut String,
        db: &dyn HirDatabase,
        span_formatter: impl Fn(FileId, TextRange) -> String,
        display_target: DisplayTarget,
    ) -> std::result::Result<(), std::fmt::Error> {
        match self {
            MirLowerError::ConstEvalError(name, e) => {
                writeln!(f, "In evaluating constant {name}")?;
                match &**e {
                    ConstEvalError::MirLowerError(e) => {
                        e.pretty_print(f, db, span_formatter, display_target)?
                    }
                    ConstEvalError::MirEvalError(e) => {
                        e.pretty_print(f, db, span_formatter, display_target)?
                    }
                }
            }
            MirLowerError::MissingFunctionDefinition(owner, it) => {
                let owner = owner.expression_store_owner(db);
                let store = ExpressionStore::of(db, owner);
                writeln!(
                    f,
                    "Missing function definition for {}",
                    hir_def::expr_store::pretty::print_expr_hir(
                        db,
                        store,
                        owner,
                        *it,
                        display_target.edition
                    )
                )?;
            }
            MirLowerError::HasErrors => writeln!(f, "Type inference result contains errors")?,
            MirLowerError::GenericArgNotProvided(id, subst) => {
                let param_name = match *id {
                    GenericParamId::TypeParamId(id) => {
                        GenericParams::of(db, id.parent())[id.local_id()].name().cloned()
                    }
                    GenericParamId::ConstParamId(id) => {
                        GenericParams::of(db, id.parent())[id.local_id()].name().cloned()
                    }
                    GenericParamId::LifetimeParamId(id) => {
                        Some(GenericParams::of(db, id.parent)[id.local_id].name.clone())
                    }
                };
                writeln!(
                    f,
                    "Generic arg not provided for {}",
                    param_name.unwrap_or(Name::missing()).display(db, display_target.edition)
                )?;
                writeln!(f, "Provided args: [")?;
                for g in subst.as_ref() {
                    write!(f, "    {},", g.display(db, display_target))?;
                }
                writeln!(f, "]")?;
            }
            MirLowerError::LayoutError(_)
            | MirLowerError::UnsizedTemporary(_)
            | MirLowerError::IncompleteExpr
            | MirLowerError::IncompletePattern
            | MirLowerError::InaccessibleLocal
            | MirLowerError::TraitFunctionDefinition(_, _)
            | MirLowerError::UnresolvedName { .. }
            | MirLowerError::RecordLiteralWithoutPath
            | MirLowerError::UnresolvedMethod(_)
            | MirLowerError::UnresolvedField
            | MirLowerError::TypeError(_)
            | MirLowerError::NotSupported(_)
            | MirLowerError::ContinueWithoutLoop
            | MirLowerError::BreakWithoutLoop
            | MirLowerError::Loop
            | MirLowerError::ImplementationError(_)
            | MirLowerError::LangItemNotFound
            | MirLowerError::MutatingRvalue
            | MirLowerError::UnresolvedLabel
            | MirLowerError::UnresolvedUpvar(_) => writeln!(f, "{self:?}")?,
        }
        Ok(())
    }
}

#[extension(pub trait MirEvalErrorPretty)]
impl MirEvalError<'_> {
    fn pretty_print(
        &self,
        f: &mut String,
        db: &dyn HirDatabase,
        span_formatter: impl Fn(FileId, TextRange) -> String,
        display_target: DisplayTarget,
    ) -> std::result::Result<(), std::fmt::Error> {
        writeln!(f, "Mir eval error:")?;
        let mut err = self;
        while let MirEvalError::InFunction(e, stack) = err {
            err = e;
            for (func, span, def) in stack.iter().take(30).rev() {
                match func {
                    Either::Left(func) => {
                        let function_name = FunctionSignature::of(db, *func);
                        writeln!(
                            f,
                            "In function {} ({:?})",
                            function_name.name.display(db, display_target.edition),
                            func
                        )?;
                    }
                    Either::Right(closure) => {
                        writeln!(f, "In {closure:?}")?;
                    }
                }
                let (source_map, self_param_syntax) = match *def {
                    InferBodyId::DefWithBodyId(def) => {
                        let body = &Body::with_source_map(db, def).1;
                        (&**body, body.self_param_syntax())
                    }
                    InferBodyId::AnonConstId(def) => {
                        let store = ExpressionStore::with_source_map(db, def.loc(db).owner).1;
                        (store, None)
                    }
                };
                let span: InFile<SyntaxNodePtr> = match *span {
                    MirSpan::ExprId(e) => match source_map.expr_syntax(e) {
                        Ok(s) => s.map(|it| it.into()),
                        Err(_) => continue,
                    },
                    MirSpan::PatId(p) => match source_map.pat_syntax(p) {
                        Ok(s) => s.map(|it| it.syntax_node_ptr()),
                        Err(_) => continue,
                    },
                    MirSpan::BindingId(b) => {
                        match source_map
                            .patterns_for_binding(b)
                            .iter()
                            .find_map(|p| source_map.pat_syntax(*p).ok())
                        {
                            Some(s) => s.map(|it| it.syntax_node_ptr()),
                            None => continue,
                        }
                    }
                    MirSpan::SelfParam => match self_param_syntax {
                        Some(s) => s.map(|it| it.syntax_node_ptr()),
                        None => continue,
                    },
                    MirSpan::Unknown => continue,
                };
                let file_id = span.file_id.original_file(db);
                let text_range = span.value.text_range();
                writeln!(f, "{}", span_formatter(file_id.file_id(db), text_range))?;
            }
        }
        match err {
            MirEvalError::InFunction(..) => unreachable!(),
            MirEvalError::LayoutError(err, ty) => {
                write!(
                    f,
                    "Layout for type `{}` is not available due {err:?}",
                    ty.as_ref()
                        .display(db, display_target)
                        .with_closure_style(ClosureStyle::ClosureWithId)
                )?;
            }
            MirEvalError::MirLowerError(func, err) => {
                let function_name = FunctionSignature::of(db, *func);
                let self_ = match func.lookup(db).container {
                    ItemContainerId::ImplId(impl_id) => Some({
                        db.impl_self_ty(impl_id)
                            .instantiate_identity()
                            .skip_norm_wip()
                            .display(db, display_target)
                            .to_string()
                    }),
                    ItemContainerId::TraitId(it) => Some(
                        TraitSignature::of(db, it)
                            .name
                            .display(db, display_target.edition)
                            .to_string(),
                    ),
                    _ => None,
                };
                writeln!(
                    f,
                    "MIR lowering for function `{}{}{}` ({:?}) failed due:",
                    self_.as_deref().unwrap_or_default(),
                    if self_.is_some() { "::" } else { "" },
                    function_name.name.display(db, display_target.edition),
                    func
                )?;
                err.pretty_print(f, db, span_formatter, display_target)?;
            }
            MirEvalError::ConstEvalError(name, err) => {
                MirLowerError::ConstEvalError((**name).into(), err.clone()).pretty_print(
                    f,
                    db,
                    span_formatter,
                    display_target,
                )?;
            }
            MirEvalError::UndefinedBehavior(_)
            | MirEvalError::TargetDataLayoutNotAvailable(_)
            | MirEvalError::Panic(_)
            | MirEvalError::MirLowerErrorForClosure(_, _)
            | MirEvalError::TypeIsUnsized(_, _)
            | MirEvalError::NotSupported(_)
            | MirEvalError::InvalidConst
            | MirEvalError::ExecutionLimitExceeded
            | MirEvalError::StackOverflow
            | MirEvalError::CoerceUnsizedError(_)
            | MirEvalError::InternalError(_)
            | MirEvalError::InvalidVTableId(_) => writeln!(f, "{err:?}")?,
        }
        Ok(())
    }

    fn is_panic(&self) -> Option<&str> {
        let mut err = self;
        while let MirEvalError::InFunction(e, _) = err {
            err = e;
        }
        match err {
            MirEvalError::Panic(msg) => Some(msg),
            _ => None,
        }
    }
}

#[extension(pub trait ConstEvalErrorPretty)]
impl ConstEvalError<'_> {
    fn pretty_print(
        &self,
        f: &mut String,
        db: &dyn HirDatabase,
        span_formatter: impl Fn(span::FileId, span::TextRange) -> String,
        display_target: DisplayTarget,
    ) -> std::result::Result<(), std::fmt::Error> {
        match self {
            ConstEvalError::MirLowerError(e) => {
                e.pretty_print(f, db, span_formatter, display_target)
            }
            ConstEvalError::MirEvalError(e) => {
                e.pretty_print(f, db, span_formatter, display_target)
            }
        }
    }
}
