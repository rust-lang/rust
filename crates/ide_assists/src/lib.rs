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

use hir::Semantics;
use ide_db::base_db::FileRange;
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
    /// Computing source change sometimes is much more costly then computing the
    /// other fields. Additionally, the actual change is not required to show
    /// the lightbulb UI, it only is needed when the user tries to apply an
    /// assist. So, we compute it lazily: the API allow requesting assists with
    /// or without source change. We could (and in fact, used to) distinguish
    /// between resolved and unresolved assists at the type level, but this is
    /// cumbersome, especially if you want to embed an assist into another data
    /// structure, such as a diagnostic.
    pub source_change: Option<SourceChange>,
}

impl Assist {
    /// Return all the assists applicable at the given position.
    pub fn get(
        db: &RootDatabase,
        config: &AssistConfig,
        resolve: bool,
        range: FileRange,
    ) -> Vec<Assist> {
        let sema = Semantics::new(db);
        let ctx = AssistContext::new(sema, config, range);
        let mut acc = Assists::new(&ctx, resolve);
        handlers::all().iter().for_each(|handler| {
            handler(&mut acc, &ctx);
        });
        acc.finish()
    }
}

mod handlers {
    use crate::{AssistContext, Assists};

    pub(crate) type Handler = fn(&mut Assists, &AssistContext) -> Option<()>;

    mod add_explicit_type;
    mod add_lifetime_to_type;
    mod add_missing_impl_members;
    mod add_turbo_fish;
    mod apply_demorgan;
    mod auto_import;
    mod change_visibility;
    mod convert_integer_literal;
    mod convert_comment_block;
    mod convert_iter_for_each_to_for;
    mod early_return;
    mod expand_glob_import;
    mod extract_function;
    mod extract_struct_from_enum_variant;
    mod extract_type_alias;
    mod extract_variable;
    mod fill_match_arms;
    mod fix_visibility;
    mod flip_binexpr;
    mod flip_comma;
    mod flip_trait_bound;
    mod generate_default_from_enum_variant;
    mod generate_default_from_new;
    mod generate_is_empty_from_len;
    mod generate_derive;
    mod generate_enum_is_method;
    mod generate_enum_projection_method;
    mod generate_from_impl_for_enum;
    mod generate_function;
    mod generate_getter_mut;
    mod generate_getter;
    mod generate_impl;
    mod generate_new;
    mod generate_setter;
    mod infer_function_return_type;
    mod inline_function;
    mod inline_local_variable;
    mod introduce_named_lifetime;
    mod invert_if;
    mod merge_imports;
    mod merge_match_arms;
    mod move_bounds;
    mod move_guard;
    mod move_module_to_file;
    mod pull_assignment_up;
    mod qualify_path;
    mod raw_string;
    mod remove_dbg;
    mod remove_mut;
    mod remove_unused_param;
    mod reorder_fields;
    mod reorder_impl;
    mod replace_derive_with_manual_impl;
    mod replace_for_loop_with_for_each;
    mod replace_if_let_with_match;
    mod replace_impl_trait_with_generic;
    mod replace_let_with_if_let;
    mod replace_qualified_name_with_use;
    mod replace_string_with_char;
    mod replace_unwrap_with_match;
    mod split_import;
    mod toggle_ignore;
    mod unmerge_use;
    mod unwrap_block;
    mod wrap_return_type_in_result;

    pub(crate) fn all() -> &'static [Handler] {
        &[
            // These are alphabetic for the foolish consistency
            add_explicit_type::add_explicit_type,
            add_lifetime_to_type::add_lifetime_to_type,
            add_turbo_fish::add_turbo_fish,
            apply_demorgan::apply_demorgan,
            auto_import::auto_import,
            change_visibility::change_visibility,
            convert_integer_literal::convert_integer_literal,
            convert_comment_block::convert_comment_block,
            convert_iter_for_each_to_for::convert_iter_for_each_to_for,
            early_return::convert_to_guarded_return,
            expand_glob_import::expand_glob_import,
            extract_struct_from_enum_variant::extract_struct_from_enum_variant,
            extract_type_alias::extract_type_alias,
            fill_match_arms::fill_match_arms,
            fix_visibility::fix_visibility,
            flip_binexpr::flip_binexpr,
            flip_comma::flip_comma,
            flip_trait_bound::flip_trait_bound,
            generate_default_from_enum_variant::generate_default_from_enum_variant,
            generate_default_from_new::generate_default_from_new,
            generate_is_empty_from_len::generate_is_empty_from_len,
            generate_derive::generate_derive,
            generate_enum_is_method::generate_enum_is_method,
            generate_enum_projection_method::generate_enum_as_method,
            generate_enum_projection_method::generate_enum_try_into_method,
            generate_from_impl_for_enum::generate_from_impl_for_enum,
            generate_function::generate_function,
            generate_getter_mut::generate_getter_mut,
            generate_getter::generate_getter,
            generate_impl::generate_impl,
            generate_new::generate_new,
            generate_setter::generate_setter,
            infer_function_return_type::infer_function_return_type,
            inline_function::inline_function,
            inline_local_variable::inline_local_variable,
            introduce_named_lifetime::introduce_named_lifetime,
            invert_if::invert_if,
            merge_imports::merge_imports,
            merge_match_arms::merge_match_arms,
            move_bounds::move_bounds_to_where_clause,
            move_guard::move_arm_cond_to_match_guard,
            move_guard::move_guard_to_arm_body,
            move_module_to_file::move_module_to_file,
            pull_assignment_up::pull_assignment_up,
            qualify_path::qualify_path,
            raw_string::add_hash,
            raw_string::make_usual_string,
            raw_string::remove_hash,
            remove_dbg::remove_dbg,
            remove_mut::remove_mut,
            remove_unused_param::remove_unused_param,
            reorder_fields::reorder_fields,
            reorder_impl::reorder_impl,
            replace_derive_with_manual_impl::replace_derive_with_manual_impl,
            replace_for_loop_with_for_each::replace_for_loop_with_for_each,
            replace_if_let_with_match::replace_if_let_with_match,
            replace_if_let_with_match::replace_match_with_if_let,
            replace_impl_trait_with_generic::replace_impl_trait_with_generic,
            replace_let_with_if_let::replace_let_with_if_let,
            replace_qualified_name_with_use::replace_qualified_name_with_use,
            replace_unwrap_with_match::replace_unwrap_with_match,
            split_import::split_import,
            toggle_ignore::toggle_ignore,
            unmerge_use::unmerge_use,
            unwrap_block::unwrap_block,
            wrap_return_type_in_result::wrap_return_type_in_result,
            // These are manually sorted for better priorities. By default,
            // priority is determined by the size of the target range (smaller
            // target wins). If the ranges are equal, position in this list is
            // used as a tie-breaker.
            add_missing_impl_members::add_missing_impl_members,
            add_missing_impl_members::add_missing_default_members,
            //
            replace_string_with_char::replace_string_with_char,
            raw_string::make_raw_string,
            //
            extract_variable::extract_variable,
            extract_function::extract_function,
            // Are you sure you want to add new assist here, and not to the
            // sorted list above?
        ]
    }
}
