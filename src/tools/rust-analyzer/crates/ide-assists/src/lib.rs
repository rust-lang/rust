//! `assists` crate provides a bunch of code assists, also known as code actions
//! (in LSP) or intentions (in IntelliJ).
//!
//! An assist is a micro-refactoring, which is automatically activated in
//! certain context. For example, if the cursor is over `,`, a "swap `,`" assist
//! becomes available.
//!
//! ## Assists Guidelines
//!
//! Assists are the main mechanism to deliver advanced IDE features to the user,
//! so we should pay extra attention to the UX.
//!
//! The power of assists comes from their context-awareness. The main problem
//! with IDE features is that there are a lot of them, and it's hard to teach
//! the user what's available. Assists solve this problem nicely: 💡 signifies
//! that *something* is possible, and clicking on it reveals a *short* list of
//! actions. Contrast it with Emacs `M-x`, which just spits an infinite list of
//! all the features.
//!
//! Here are some considerations when creating a new assist:
//!
//! * It's good to preserve semantics, and it's good to keep the code compiling,
//!   but it isn't necessary. Example: "flip binary operation" might change
//!   semantics.
//! * Assist shouldn't necessary make the code "better". A lot of assist come in
//!   pairs: "if let <-> match".
//! * Assists should have as narrow scope as possible. Each new assists greatly
//!   improves UX for cases where the user actually invokes it, but it makes UX
//!   worse for every case where the user clicks 💡 to invoke some *other*
//!   assist. So, a rarely useful assist which is always applicable can be a net
//!   negative.
//! * Rarely useful actions are tricky. Sometimes there are features which are
//!   clearly useful to some users, but are just noise most of the time. We
//!   don't have a good solution here, our current approach is to make this
//!   functionality available only if assist is applicable to the whole
//!   selection. Example: `sort_items` sorts items alphabetically. Naively, it
//!   should be available more or less everywhere, which isn't useful. So
//!   instead we only show it if the user *selects* the items they want to sort.
//! * Consider grouping related assists together (see [`Assists::add_group`]).
//! * Make assists robust. If the assist depends on results of type-inference too
//!   much, it might only fire in fully-correct code. This makes assist less
//!   useful and (worse) less predictable. The user should have a clear
//!   intuition when each particular assist is available.
//! * Make small assists, which compose. Example: rather than auto-importing
//!   enums in `add_missing_match_arms`, we use fully-qualified names. There's a
//!   separate assist to shorten a fully-qualified name.
//! * Distinguish between assists and fixits for diagnostics. Internally, fixits
//!   and assists are equivalent. They have the same "show a list + invoke a
//!   single element" workflow, and both use [`Assist`] data structure. The main
//!   difference is in the UX: while 💡 looks only at the cursor position,
//!   diagnostics squigglies and fixits are calculated for the whole file and
//!   are presented to the user eagerly. So, diagnostics should be fixable
//!   errors, while assists can be just suggestions for an alternative way to do
//!   something. If something *could* be a diagnostic, it should be a
//!   diagnostic. Conversely, it might be valuable to turn a diagnostic with a
//!   lot of false errors into an assist.
//!
//! See also this post:
//! <https://rust-analyzer.github.io/blog/2020/09/28/how-to-make-a-light-bulb.html>

#![warn(rust_2018_idioms, unused_lifetimes, semicolon_in_expressions_from_macros)]

#[allow(unused)]
macro_rules! eprintln {
    ($($tt:tt)*) => { stdx::eprintln!($($tt)*) };
}

mod assist_config;
mod assist_context;
#[cfg(test)]
mod tests;
pub mod utils;

use hir::Semantics;
use ide_db::{base_db::FileRange, RootDatabase};
use syntax::TextRange;

pub(crate) use crate::assist_context::{AssistContext, Assists};

pub use assist_config::AssistConfig;
pub use ide_db::assists::{
    Assist, AssistId, AssistKind, AssistResolveStrategy, GroupLabel, SingleResolve,
};

/// Return all the assists applicable at the given position.
///
// NOTE: We don't have a `Feature: ` section for assists, they are special-cased
// in the manual.
pub fn assists(
    db: &RootDatabase,
    config: &AssistConfig,
    resolve: AssistResolveStrategy,
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

mod handlers {
    use crate::{AssistContext, Assists};

    pub(crate) type Handler = fn(&mut Assists, &AssistContext<'_>) -> Option<()>;

    mod add_braces;
    mod add_explicit_type;
    mod add_label_to_loop;
    mod add_lifetime_to_type;
    mod add_missing_impl_members;
    mod add_turbo_fish;
    mod apply_demorgan;
    mod auto_import;
    mod change_visibility;
    mod convert_bool_then;
    mod convert_comment_block;
    mod convert_integer_literal;
    mod convert_into_to_from;
    mod convert_iter_for_each_to_for;
    mod convert_let_else_to_match;
    mod convert_match_to_let_else;
    mod convert_nested_function_to_closure;
    mod convert_tuple_struct_to_named_struct;
    mod convert_named_struct_to_tuple_struct;
    mod convert_to_guarded_return;
    mod convert_two_arm_bool_match_to_matches_macro;
    mod convert_while_to_loop;
    mod desugar_doc_comment;
    mod destructure_tuple_binding;
    mod expand_glob_import;
    mod extract_expressions_from_format_string;
    mod extract_function;
    mod extract_module;
    mod extract_struct_from_enum_variant;
    mod extract_type_alias;
    mod extract_variable;
    mod add_missing_match_arms;
    mod fix_visibility;
    mod flip_binexpr;
    mod flip_comma;
    mod flip_trait_bound;
    mod generate_constant;
    mod generate_default_from_enum_variant;
    mod generate_default_from_new;
    mod generate_delegate_trait;
    mod generate_deref;
    mod generate_derive;
    mod generate_documentation_template;
    mod generate_enum_is_method;
    mod generate_enum_projection_method;
    mod generate_enum_variant;
    mod generate_from_impl_for_enum;
    mod generate_function;
    mod generate_getter_or_setter;
    mod generate_impl;
    mod generate_is_empty_from_len;
    mod generate_new;
    mod generate_delegate_methods;
    mod generate_trait_from_impl;
    mod add_return_type;
    mod inline_call;
    mod inline_const_as_literal;
    mod inline_local_variable;
    mod inline_macro;
    mod inline_type_alias;
    mod introduce_named_lifetime;
    mod invert_if;
    mod merge_imports;
    mod merge_match_arms;
    mod move_bounds;
    mod move_const_to_impl;
    mod move_guard;
    mod move_module_to_file;
    mod move_to_mod_rs;
    mod move_from_mod_rs;
    mod number_representation;
    mod promote_local_to_const;
    mod pull_assignment_up;
    mod qualify_path;
    mod qualify_method_call;
    mod raw_string;
    mod remove_dbg;
    mod remove_mut;
    mod remove_unused_param;
    mod remove_parentheses;
    mod reorder_fields;
    mod reorder_impl_items;
    mod replace_try_expr_with_match;
    mod replace_derive_with_manual_impl;
    mod replace_if_let_with_match;
    mod replace_method_eager_lazy;
    mod replace_arith_op;
    mod introduce_named_generic;
    mod replace_let_with_if_let;
    mod replace_named_generic_with_impl;
    mod replace_qualified_name_with_use;
    mod replace_string_with_char;
    mod replace_turbofish_with_explicit_type;
    mod split_import;
    mod unmerge_match_arm;
    mod unwrap_tuple;
    mod sort_items;
    mod toggle_ignore;
    mod unmerge_use;
    mod unnecessary_async;
    mod unwrap_block;
    mod unwrap_result_return_type;
    mod unqualify_method_call;
    mod wrap_return_type_in_result;

    pub(crate) fn all() -> &'static [Handler] {
        &[
            // These are alphabetic for the foolish consistency
            add_braces::add_braces,
            add_explicit_type::add_explicit_type,
            add_label_to_loop::add_label_to_loop,
            add_missing_match_arms::add_missing_match_arms,
            add_lifetime_to_type::add_lifetime_to_type,
            add_return_type::add_return_type,
            add_turbo_fish::add_turbo_fish,
            apply_demorgan::apply_demorgan,
            auto_import::auto_import,
            change_visibility::change_visibility,
            convert_bool_then::convert_bool_then_to_if,
            convert_bool_then::convert_if_to_bool_then,
            convert_comment_block::convert_comment_block,
            convert_integer_literal::convert_integer_literal,
            convert_into_to_from::convert_into_to_from,
            convert_iter_for_each_to_for::convert_iter_for_each_to_for,
            convert_iter_for_each_to_for::convert_for_loop_with_for_each,
            convert_let_else_to_match::convert_let_else_to_match,
            convert_match_to_let_else::convert_match_to_let_else,
            convert_named_struct_to_tuple_struct::convert_named_struct_to_tuple_struct,
            convert_nested_function_to_closure::convert_nested_function_to_closure,
            convert_to_guarded_return::convert_to_guarded_return,
            convert_tuple_struct_to_named_struct::convert_tuple_struct_to_named_struct,
            convert_two_arm_bool_match_to_matches_macro::convert_two_arm_bool_match_to_matches_macro,
            convert_while_to_loop::convert_while_to_loop,
            desugar_doc_comment::desugar_doc_comment,
            destructure_tuple_binding::destructure_tuple_binding,
            expand_glob_import::expand_glob_import,
            extract_expressions_from_format_string::extract_expressions_from_format_string,
            extract_struct_from_enum_variant::extract_struct_from_enum_variant,
            extract_type_alias::extract_type_alias,
            fix_visibility::fix_visibility,
            flip_binexpr::flip_binexpr,
            flip_comma::flip_comma,
            flip_trait_bound::flip_trait_bound,
            generate_constant::generate_constant,
            generate_default_from_enum_variant::generate_default_from_enum_variant,
            generate_default_from_new::generate_default_from_new,
            generate_delegate_trait::generate_delegate_trait,
            generate_derive::generate_derive,
            generate_documentation_template::generate_documentation_template,
            generate_documentation_template::generate_doc_example,
            generate_enum_is_method::generate_enum_is_method,
            generate_enum_projection_method::generate_enum_as_method,
            generate_enum_projection_method::generate_enum_try_into_method,
            generate_enum_variant::generate_enum_variant,
            generate_from_impl_for_enum::generate_from_impl_for_enum,
            generate_function::generate_function,
            generate_impl::generate_impl,
            generate_impl::generate_trait_impl,
            generate_is_empty_from_len::generate_is_empty_from_len,
            generate_new::generate_new,
            generate_trait_from_impl::generate_trait_from_impl,
            inline_call::inline_call,
            inline_call::inline_into_callers,
            inline_const_as_literal::inline_const_as_literal,
            inline_local_variable::inline_local_variable,
            inline_type_alias::inline_type_alias,
            inline_type_alias::inline_type_alias_uses,
            introduce_named_generic::introduce_named_generic,
            introduce_named_lifetime::introduce_named_lifetime,
            invert_if::invert_if,
            merge_imports::merge_imports,
            merge_match_arms::merge_match_arms,
            move_bounds::move_bounds_to_where_clause,
            move_const_to_impl::move_const_to_impl,
            move_guard::move_arm_cond_to_match_guard,
            move_guard::move_guard_to_arm_body,
            move_module_to_file::move_module_to_file,
            move_to_mod_rs::move_to_mod_rs,
            move_from_mod_rs::move_from_mod_rs,
            number_representation::reformat_number_literal,
            pull_assignment_up::pull_assignment_up,
            promote_local_to_const::promote_local_to_const,
            qualify_path::qualify_path,
            qualify_method_call::qualify_method_call,
            raw_string::add_hash,
            raw_string::make_usual_string,
            raw_string::remove_hash,
            remove_mut::remove_mut,
            remove_unused_param::remove_unused_param,
            remove_parentheses::remove_parentheses,
            reorder_fields::reorder_fields,
            reorder_impl_items::reorder_impl_items,
            replace_try_expr_with_match::replace_try_expr_with_match,
            replace_derive_with_manual_impl::replace_derive_with_manual_impl,
            replace_if_let_with_match::replace_if_let_with_match,
            replace_if_let_with_match::replace_match_with_if_let,
            replace_let_with_if_let::replace_let_with_if_let,
            replace_method_eager_lazy::replace_with_eager_method,
            replace_method_eager_lazy::replace_with_lazy_method,
            replace_named_generic_with_impl::replace_named_generic_with_impl,
            replace_turbofish_with_explicit_type::replace_turbofish_with_explicit_type,
            replace_qualified_name_with_use::replace_qualified_name_with_use,
            replace_arith_op::replace_arith_with_wrapping,
            replace_arith_op::replace_arith_with_checked,
            replace_arith_op::replace_arith_with_saturating,
            sort_items::sort_items,
            split_import::split_import,
            toggle_ignore::toggle_ignore,
            unmerge_match_arm::unmerge_match_arm,
            unmerge_use::unmerge_use,
            unnecessary_async::unnecessary_async,
            unwrap_block::unwrap_block,
            unwrap_result_return_type::unwrap_result_return_type,
            unwrap_tuple::unwrap_tuple,
            unqualify_method_call::unqualify_method_call,
            wrap_return_type_in_result::wrap_return_type_in_result,
            // These are manually sorted for better priorities. By default,
            // priority is determined by the size of the target range (smaller
            // target wins). If the ranges are equal, position in this list is
            // used as a tie-breaker.
            add_missing_impl_members::add_missing_impl_members,
            add_missing_impl_members::add_missing_default_members,
            //
            replace_string_with_char::replace_string_with_char,
            replace_string_with_char::replace_char_with_string,
            raw_string::make_raw_string,
            //
            extract_variable::extract_variable,
            extract_function::extract_function,
            extract_module::extract_module,
            //
            generate_getter_or_setter::generate_getter,
            generate_getter_or_setter::generate_getter_mut,
            generate_getter_or_setter::generate_setter,
            generate_delegate_methods::generate_delegate_methods,
            generate_deref::generate_deref,
            //
            remove_dbg::remove_dbg,
            inline_macro::inline_macro,
            // Are you sure you want to add new assist here, and not to the
            // sorted list above?
        ]
    }
}
