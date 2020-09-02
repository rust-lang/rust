//! `assists` crate provides a bunch of code assists, also known as code
//! actions (in LSP) or intentions (in IntelliJ).
//!
//! An assist is a micro-refactoring, which is automatically activated in
//! certain context. For example, if the cursor is over `,`, a "swap `,`" assist
//! becomes available.

#[allow(unused)]
macro_rules! eprintln {
    ($($tt:tt)*) => { stdx::eprintln!($($tt)*) };
}

mod assist_config;
mod assist_context;
#[cfg(test)]
mod tests;
pub mod utils;
pub mod ast_transform;

use base_db::FileRange;
use hir::Semantics;
use ide_db::{label::Label, source_change::SourceChange, RootDatabase};
use syntax::TextRange;

pub(crate) use crate::assist_context::{AssistContext, Assists};

pub use assist_config::AssistConfig;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AssistKind {
    None,
    QuickFix,
    Generate,
    Refactor,
    RefactorExtract,
    RefactorInline,
    RefactorRewrite,
}

impl AssistKind {
    pub fn contains(self, other: AssistKind) -> bool {
        if self == other {
            return true;
        }

        match self {
            AssistKind::None | AssistKind::Generate => return true,
            AssistKind::Refactor => match other {
                AssistKind::RefactorExtract
                | AssistKind::RefactorInline
                | AssistKind::RefactorRewrite => return true,
                _ => return false,
            },
            _ => return false,
        }
    }
}

/// Unique identifier of the assist, should not be shown to the user
/// directly.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct AssistId(pub &'static str, pub AssistKind);

#[derive(Clone, Debug)]
pub struct GroupLabel(pub String);

#[derive(Debug, Clone)]
pub struct Assist {
    pub id: AssistId,
    /// Short description of the assist, as shown in the UI.
    pub label: Label,
    pub group: Option<GroupLabel>,
    /// Target ranges are used to sort assists: the smaller the target range,
    /// the more specific assist is, and so it should be sorted first.
    pub target: TextRange,
}

#[derive(Debug, Clone)]
pub struct ResolvedAssist {
    pub assist: Assist,
    pub source_change: SourceChange,
}

impl Assist {
    /// Return all the assists applicable at the given position.
    ///
    /// Assists are returned in the "unresolved" state, that is only labels are
    /// returned, without actual edits.
    pub fn unresolved(db: &RootDatabase, config: &AssistConfig, range: FileRange) -> Vec<Assist> {
        let sema = Semantics::new(db);
        let ctx = AssistContext::new(sema, config, range);
        let mut acc = Assists::new_unresolved(&ctx);
        handlers::all().iter().for_each(|handler| {
            handler(&mut acc, &ctx);
        });
        acc.finish_unresolved()
    }

    /// Return all the assists applicable at the given position.
    ///
    /// Assists are returned in the "resolved" state, that is with edit fully
    /// computed.
    pub fn resolved(
        db: &RootDatabase,
        config: &AssistConfig,
        range: FileRange,
    ) -> Vec<ResolvedAssist> {
        let sema = Semantics::new(db);
        let ctx = AssistContext::new(sema, config, range);
        let mut acc = Assists::new_resolved(&ctx);
        handlers::all().iter().for_each(|handler| {
            handler(&mut acc, &ctx);
        });
        acc.finish_resolved()
    }
}

mod handlers {
    use crate::{AssistContext, Assists};

    pub(crate) type Handler = fn(&mut Assists, &AssistContext) -> Option<()>;

    mod add_custom_impl;
    mod add_explicit_type;
    mod add_missing_impl_members;
    mod add_turbo_fish;
    mod apply_demorgan;
    mod auto_import;
    mod change_return_type_to_result;
    mod change_visibility;
    mod early_return;
    mod expand_glob_import;
    mod extract_struct_from_enum_variant;
    mod extract_variable;
    mod fill_match_arms;
    mod fix_visibility;
    mod flip_binexpr;
    mod flip_comma;
    mod flip_trait_bound;
    mod generate_derive;
    mod generate_from_impl_for_enum;
    mod generate_function;
    mod generate_impl;
    mod generate_new;
    mod inline_local_variable;
    mod introduce_named_lifetime;
    mod invert_if;
    mod merge_imports;
    mod merge_match_arms;
    mod move_bounds;
    mod move_guard;
    mod raw_string;
    mod remove_dbg;
    mod remove_mut;
    mod remove_unused_param;
    mod reorder_fields;
    mod replace_if_let_with_match;
    mod replace_impl_trait_with_generic;
    mod replace_let_with_if_let;
    mod replace_qualified_name_with_use;
    mod replace_unwrap_with_match;
    mod split_import;
    mod unwrap_block;

    pub(crate) fn all() -> &'static [Handler] {
        &[
            // These are alphabetic for the foolish consistency
            add_custom_impl::add_custom_impl,
            add_explicit_type::add_explicit_type,
            add_turbo_fish::add_turbo_fish,
            apply_demorgan::apply_demorgan,
            auto_import::auto_import,
            change_return_type_to_result::change_return_type_to_result,
            change_visibility::change_visibility,
            early_return::convert_to_guarded_return,
            expand_glob_import::expand_glob_import,
            extract_struct_from_enum_variant::extract_struct_from_enum_variant,
            extract_variable::extract_variable,
            fill_match_arms::fill_match_arms,
            fix_visibility::fix_visibility,
            flip_binexpr::flip_binexpr,
            flip_comma::flip_comma,
            flip_trait_bound::flip_trait_bound,
            generate_derive::generate_derive,
            generate_from_impl_for_enum::generate_from_impl_for_enum,
            generate_function::generate_function,
            generate_impl::generate_impl,
            generate_new::generate_new,
            inline_local_variable::inline_local_variable,
            introduce_named_lifetime::introduce_named_lifetime,
            invert_if::invert_if,
            merge_imports::merge_imports,
            merge_match_arms::merge_match_arms,
            move_bounds::move_bounds_to_where_clause,
            move_guard::move_arm_cond_to_match_guard,
            move_guard::move_guard_to_arm_body,
            raw_string::add_hash,
            raw_string::make_raw_string,
            raw_string::make_usual_string,
            raw_string::remove_hash,
            remove_dbg::remove_dbg,
            remove_mut::remove_mut,
            remove_unused_param::remove_unused_param,
            reorder_fields::reorder_fields,
            replace_if_let_with_match::replace_if_let_with_match,
            replace_impl_trait_with_generic::replace_impl_trait_with_generic,
            replace_let_with_if_let::replace_let_with_if_let,
            replace_qualified_name_with_use::replace_qualified_name_with_use,
            replace_unwrap_with_match::replace_unwrap_with_match,
            split_import::split_import,
            unwrap_block::unwrap_block,
            // These are manually sorted for better priorities
            add_missing_impl_members::add_missing_impl_members,
            add_missing_impl_members::add_missing_default_members,
            // Are you sure you want to add new assist here, and not to the
            // sorted list above?
        ]
    }
}
