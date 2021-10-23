//! Rewrite a list some items with overflow.

use std::cmp::min;

use itertools::Itertools;
use rustc_ast::token::DelimToken;
use rustc_ast::{ast, ptr};
use rustc_span::Span;

use crate::closures;
use crate::config::lists::*;
use crate::config::Version;
use crate::expr::{
    can_be_overflowed_expr, is_every_expr_simple, is_method_call, is_nested_call, is_simple_expr,
    rewrite_cond,
};
use crate::lists::{
    definitive_tactic, itemize_list, write_list, ListFormatting, ListItem, Separator,
};
use crate::macros::MacroArg;
use crate::patterns::{can_be_overflowed_pat, TuplePatField};
use crate::rewrite::{Rewrite, RewriteContext};
use crate::shape::Shape;
use crate::source_map::SpanUtils;
use crate::spanned::Spanned;
use crate::types::{can_be_overflowed_type, SegmentParam};
use crate::utils::{count_newlines, extra_offset, first_line_width, last_line_width, mk_sp};

const SHORT_ITEM_THRESHOLD: usize = 10;

/// A list of `format!`-like macros, that take a long format string and a list of arguments to
/// format.
///
/// Organized as a list of `(&str, usize)` tuples, giving the name of the macro and the number of
/// arguments before the format string (none for `format!("format", ...)`, one for `assert!(result,
/// "format", ...)`, two for `assert_eq!(left, right, "format", ...)`).
const SPECIAL_MACRO_WHITELIST: &[(&str, usize)] = &[
    // format! like macros
    // From the Rust Standard Library.
    ("eprint!", 0),
    ("eprintln!", 0),
    ("format!", 0),
    ("format_args!", 0),
    ("print!", 0),
    ("println!", 0),
    ("panic!", 0),
    ("unreachable!", 0),
    // From the `log` crate.
    ("debug!", 0),
    ("error!", 0),
    ("info!", 0),
    ("warn!", 0),
    // write! like macros
    ("assert!", 1),
    ("debug_assert!", 1),
    ("write!", 1),
    ("writeln!", 1),
    // assert_eq! like macros
    ("assert_eq!", 2),
    ("assert_ne!", 2),
    ("debug_assert_eq!", 2),
    ("debug_assert_ne!", 2),
];

const SPECIAL_ATTR_WHITELIST: &[(&str, usize)] = &[
    // From the `failure` crate.
    ("fail", 0),
];

#[derive(Debug)]
pub(crate) enum OverflowableItem<'a> {
    Expr(&'a ast::Expr),
    GenericParam(&'a ast::GenericParam),
    MacroArg(&'a MacroArg),
    NestedMetaItem(&'a ast::NestedMetaItem),
    SegmentParam(&'a SegmentParam<'a>),
    FieldDef(&'a ast::FieldDef),
    TuplePatField(&'a TuplePatField<'a>),
    Ty(&'a ast::Ty),
    Pat(&'a ast::Pat),
}

impl<'a> Rewrite for OverflowableItem<'a> {
    fn rewrite(&self, context: &RewriteContext<'_>, shape: Shape) -> Option<String> {
        self.map(|item| item.rewrite(context, shape))
    }
}

impl<'a> Spanned for OverflowableItem<'a> {
    fn span(&self) -> Span {
        self.map(|item| item.span())
    }
}

impl<'a> OverflowableItem<'a> {
    fn has_attrs(&self) -> bool {
        match self {
            OverflowableItem::Expr(ast::Expr { attrs, .. })
            | OverflowableItem::GenericParam(ast::GenericParam { attrs, .. }) => !attrs.is_empty(),
            OverflowableItem::FieldDef(ast::FieldDef { attrs, .. }) => !attrs.is_empty(),
            OverflowableItem::MacroArg(MacroArg::Expr(expr)) => !expr.attrs.is_empty(),
            OverflowableItem::MacroArg(MacroArg::Item(item)) => !item.attrs.is_empty(),
            _ => false,
        }
    }

    pub(crate) fn map<F, T>(&self, f: F) -> T
    where
        F: Fn(&dyn IntoOverflowableItem<'a>) -> T,
    {
        match self {
            OverflowableItem::Expr(expr) => f(*expr),
            OverflowableItem::GenericParam(gp) => f(*gp),
            OverflowableItem::MacroArg(macro_arg) => f(*macro_arg),
            OverflowableItem::NestedMetaItem(nmi) => f(*nmi),
            OverflowableItem::SegmentParam(sp) => f(*sp),
            OverflowableItem::FieldDef(sf) => f(*sf),
            OverflowableItem::TuplePatField(pat) => f(*pat),
            OverflowableItem::Ty(ty) => f(*ty),
            OverflowableItem::Pat(pat) => f(*pat),
        }
    }

    pub(crate) fn is_simple(&self) -> bool {
        match self {
            OverflowableItem::Expr(expr) => is_simple_expr(expr),
            OverflowableItem::MacroArg(MacroArg::Keyword(..)) => true,
            OverflowableItem::MacroArg(MacroArg::Expr(expr)) => is_simple_expr(expr),
            OverflowableItem::NestedMetaItem(nested_meta_item) => match nested_meta_item {
                ast::NestedMetaItem::Literal(..) => true,
                ast::NestedMetaItem::MetaItem(ref meta_item) => {
                    matches!(meta_item.kind, ast::MetaItemKind::Word)
                }
            },
            _ => false,
        }
    }

    pub(crate) fn is_expr(&self) -> bool {
        matches!(
            self,
            OverflowableItem::Expr(..) | OverflowableItem::MacroArg(MacroArg::Expr(..))
        )
    }

    pub(crate) fn is_nested_call(&self) -> bool {
        match self {
            OverflowableItem::Expr(expr) => is_nested_call(expr),
            OverflowableItem::MacroArg(MacroArg::Expr(expr)) => is_nested_call(expr),
            _ => false,
        }
    }

    pub(crate) fn to_expr(&self) -> Option<&'a ast::Expr> {
        match self {
            OverflowableItem::Expr(expr) => Some(expr),
            OverflowableItem::MacroArg(MacroArg::Expr(ref expr)) => Some(expr),
            _ => None,
        }
    }

    pub(crate) fn can_be_overflowed(&self, context: &RewriteContext<'_>, len: usize) -> bool {
        match self {
            OverflowableItem::Expr(expr) => can_be_overflowed_expr(context, expr, len),
            OverflowableItem::MacroArg(macro_arg) => match macro_arg {
                MacroArg::Expr(ref expr) => can_be_overflowed_expr(context, expr, len),
                MacroArg::Ty(ref ty) => can_be_overflowed_type(context, ty, len),
                MacroArg::Pat(..) => false,
                MacroArg::Item(..) => len == 1,
                MacroArg::Keyword(..) => false,
            },
            OverflowableItem::NestedMetaItem(nested_meta_item) if len == 1 => {
                match nested_meta_item {
                    ast::NestedMetaItem::Literal(..) => false,
                    ast::NestedMetaItem::MetaItem(..) => true,
                }
            }
            OverflowableItem::SegmentParam(SegmentParam::Type(ty)) => {
                can_be_overflowed_type(context, ty, len)
            }
            OverflowableItem::TuplePatField(pat) => can_be_overflowed_pat(context, pat, len),
            OverflowableItem::Ty(ty) => can_be_overflowed_type(context, ty, len),
            _ => false,
        }
    }

    fn whitelist(&self) -> &'static [(&'static str, usize)] {
        match self {
            OverflowableItem::MacroArg(..) => SPECIAL_MACRO_WHITELIST,
            OverflowableItem::NestedMetaItem(..) => SPECIAL_ATTR_WHITELIST,
            _ => &[],
        }
    }
}

pub(crate) trait IntoOverflowableItem<'a>: Rewrite + Spanned {
    fn into_overflowable_item(&'a self) -> OverflowableItem<'a>;
}

impl<'a, T: 'a + IntoOverflowableItem<'a>> IntoOverflowableItem<'a> for ptr::P<T> {
    fn into_overflowable_item(&'a self) -> OverflowableItem<'a> {
        (**self).into_overflowable_item()
    }
}

macro_rules! impl_into_overflowable_item_for_ast_node {
    ($($ast_node:ident),*) => {
        $(
            impl<'a> IntoOverflowableItem<'a> for ast::$ast_node {
                fn into_overflowable_item(&'a self) -> OverflowableItem<'a> {
                    OverflowableItem::$ast_node(self)
                }
            }
        )*
    }
}

macro_rules! impl_into_overflowable_item_for_rustfmt_types {
    ([$($ty:ident),*], [$($ty_with_lifetime:ident),*]) => {
        $(
            impl<'a> IntoOverflowableItem<'a> for $ty {
                fn into_overflowable_item(&'a self) -> OverflowableItem<'a> {
                    OverflowableItem::$ty(self)
                }
            }
        )*
        $(
            impl<'a> IntoOverflowableItem<'a> for $ty_with_lifetime<'a> {
                fn into_overflowable_item(&'a self) -> OverflowableItem<'a> {
                    OverflowableItem::$ty_with_lifetime(self)
                }
            }
        )*
    }
}

impl_into_overflowable_item_for_ast_node!(Expr, GenericParam, NestedMetaItem, FieldDef, Ty, Pat);
impl_into_overflowable_item_for_rustfmt_types!([MacroArg], [SegmentParam, TuplePatField]);

pub(crate) fn into_overflowable_list<'a, T>(
    iter: impl Iterator<Item = &'a T>,
) -> impl Iterator<Item = OverflowableItem<'a>>
where
    T: 'a + IntoOverflowableItem<'a>,
{
    iter.map(|x| IntoOverflowableItem::into_overflowable_item(x))
}

pub(crate) fn rewrite_with_parens<'a, T: 'a + IntoOverflowableItem<'a>>(
    context: &'a RewriteContext<'_>,
    ident: &'a str,
    items: impl Iterator<Item = &'a T>,
    shape: Shape,
    span: Span,
    item_max_width: usize,
    force_separator_tactic: Option<SeparatorTactic>,
) -> Option<String> {
    Context::new(
        context,
        items,
        ident,
        shape,
        span,
        "(",
        ")",
        item_max_width,
        force_separator_tactic,
        None,
    )
    .rewrite(shape)
}

pub(crate) fn rewrite_with_angle_brackets<'a, T: 'a + IntoOverflowableItem<'a>>(
    context: &'a RewriteContext<'_>,
    ident: &'a str,
    items: impl Iterator<Item = &'a T>,
    shape: Shape,
    span: Span,
) -> Option<String> {
    Context::new(
        context,
        items,
        ident,
        shape,
        span,
        "<",
        ">",
        context.config.max_width(),
        None,
        None,
    )
    .rewrite(shape)
}

pub(crate) fn rewrite_with_square_brackets<'a, T: 'a + IntoOverflowableItem<'a>>(
    context: &'a RewriteContext<'_>,
    name: &'a str,
    items: impl Iterator<Item = &'a T>,
    shape: Shape,
    span: Span,
    force_separator_tactic: Option<SeparatorTactic>,
    delim_token: Option<DelimToken>,
) -> Option<String> {
    let (lhs, rhs) = match delim_token {
        Some(DelimToken::Paren) => ("(", ")"),
        Some(DelimToken::Brace) => ("{", "}"),
        _ => ("[", "]"),
    };
    Context::new(
        context,
        items,
        name,
        shape,
        span,
        lhs,
        rhs,
        context.config.array_width(),
        force_separator_tactic,
        Some(("[", "]")),
    )
    .rewrite(shape)
}

struct Context<'a> {
    context: &'a RewriteContext<'a>,
    items: Vec<OverflowableItem<'a>>,
    ident: &'a str,
    prefix: &'static str,
    suffix: &'static str,
    one_line_shape: Shape,
    nested_shape: Shape,
    span: Span,
    item_max_width: usize,
    one_line_width: usize,
    force_separator_tactic: Option<SeparatorTactic>,
    custom_delims: Option<(&'a str, &'a str)>,
}

impl<'a> Context<'a> {
    fn new<T: 'a + IntoOverflowableItem<'a>>(
        context: &'a RewriteContext<'_>,
        items: impl Iterator<Item = &'a T>,
        ident: &'a str,
        shape: Shape,
        span: Span,
        prefix: &'static str,
        suffix: &'static str,
        item_max_width: usize,
        force_separator_tactic: Option<SeparatorTactic>,
        custom_delims: Option<(&'a str, &'a str)>,
    ) -> Context<'a> {
        let used_width = extra_offset(ident, shape);
        // 1 = `()`
        let one_line_width = shape.width.saturating_sub(used_width + 2);

        // 1 = "(" or ")"
        let one_line_shape = shape
            .offset_left(last_line_width(ident) + 1)
            .and_then(|shape| shape.sub_width(1))
            .unwrap_or(Shape { width: 0, ..shape });
        let nested_shape = shape_from_indent_style(context, shape, used_width + 2, used_width + 1);
        Context {
            context,
            items: into_overflowable_list(items).collect(),
            ident,
            one_line_shape,
            nested_shape,
            span,
            prefix,
            suffix,
            item_max_width,
            one_line_width,
            force_separator_tactic,
            custom_delims,
        }
    }

    fn last_item(&self) -> Option<&OverflowableItem<'_>> {
        self.items.last()
    }

    fn items_span(&self) -> Span {
        let span_lo = self
            .context
            .snippet_provider
            .span_after(self.span, self.prefix);
        mk_sp(span_lo, self.span.hi())
    }

    fn rewrite_last_item_with_overflow(
        &self,
        last_list_item: &mut ListItem,
        shape: Shape,
    ) -> Option<String> {
        let last_item = self.last_item()?;
        let rewrite = match last_item {
            OverflowableItem::Expr(ref expr) => {
                match expr.kind {
                    // When overflowing the closure which consists of a single control flow
                    // expression, force to use block if its condition uses multi line.
                    ast::ExprKind::Closure(..) => {
                        // If the argument consists of multiple closures, we do not overflow
                        // the last closure.
                        if closures::args_have_many_closure(&self.items) {
                            None
                        } else {
                            closures::rewrite_last_closure(self.context, expr, shape)
                        }
                    }

                    // When overflowing the expressions which consists of a control flow
                    // expression, avoid condition to use multi line.
                    ast::ExprKind::If(..)
                    | ast::ExprKind::ForLoop(..)
                    | ast::ExprKind::Loop(..)
                    | ast::ExprKind::While(..)
                    | ast::ExprKind::Match(..) => {
                        let multi_line = rewrite_cond(self.context, expr, shape)
                            .map_or(false, |cond| cond.contains('\n'));

                        if multi_line {
                            None
                        } else {
                            expr.rewrite(self.context, shape)
                        }
                    }

                    _ => expr.rewrite(self.context, shape),
                }
            }
            item => item.rewrite(self.context, shape),
        };

        if let Some(rewrite) = rewrite {
            // splitn(2, *).next().unwrap() is always safe.
            let rewrite_first_line = Some(rewrite.splitn(2, '\n').next().unwrap().to_owned());
            last_list_item.item = rewrite_first_line;
            Some(rewrite)
        } else {
            None
        }
    }

    fn default_tactic(&self, list_items: &[ListItem]) -> DefinitiveListTactic {
        definitive_tactic(
            list_items,
            ListTactic::LimitedHorizontalVertical(self.item_max_width),
            Separator::Comma,
            self.one_line_width,
        )
    }

    fn try_overflow_last_item(&self, list_items: &mut Vec<ListItem>) -> DefinitiveListTactic {
        // 1 = "("
        let combine_arg_with_callee = self.items.len() == 1
            && self.items[0].is_expr()
            && !self.items[0].has_attrs()
            && self.ident.len() < self.context.config.tab_spaces();
        let overflow_last = combine_arg_with_callee || can_be_overflowed(self.context, &self.items);

        // Replace the last item with its first line to see if it fits with
        // first arguments.
        let placeholder = if overflow_last {
            let old_value = self.context.force_one_line_chain.get();
            match self.last_item() {
                Some(OverflowableItem::Expr(expr))
                    if !combine_arg_with_callee && is_method_call(expr) =>
                {
                    self.context.force_one_line_chain.replace(true);
                }
                Some(OverflowableItem::MacroArg(MacroArg::Expr(expr)))
                    if !combine_arg_with_callee
                        && is_method_call(expr)
                        && self.context.config.version() == Version::Two =>
                {
                    self.context.force_one_line_chain.replace(true);
                }
                _ => (),
            }
            let result = last_item_shape(
                &self.items,
                list_items,
                self.one_line_shape,
                self.item_max_width,
            )
            .and_then(|arg_shape| {
                self.rewrite_last_item_with_overflow(
                    &mut list_items[self.items.len() - 1],
                    arg_shape,
                )
            });
            self.context.force_one_line_chain.replace(old_value);
            result
        } else {
            None
        };

        let mut tactic = definitive_tactic(
            &*list_items,
            ListTactic::LimitedHorizontalVertical(self.item_max_width),
            Separator::Comma,
            self.one_line_width,
        );

        // Replace the stub with the full overflowing last argument if the rewrite
        // succeeded and its first line fits with the other arguments.
        match (overflow_last, tactic, placeholder) {
            (true, DefinitiveListTactic::Horizontal, Some(ref overflowed))
                if self.items.len() == 1 =>
            {
                // When we are rewriting a nested function call, we restrict the
                // budget for the inner function to avoid them being deeply nested.
                // However, when the inner function has a prefix or a suffix
                // (e.g., `foo() as u32`), this budget reduction may produce poorly
                // formatted code, where a prefix or a suffix being left on its own
                // line. Here we explicitlly check those cases.
                if count_newlines(overflowed) == 1 {
                    let rw = self
                        .items
                        .last()
                        .and_then(|last_item| last_item.rewrite(self.context, self.nested_shape));
                    let no_newline = rw.as_ref().map_or(false, |s| !s.contains('\n'));
                    if no_newline {
                        list_items[self.items.len() - 1].item = rw;
                    } else {
                        list_items[self.items.len() - 1].item = Some(overflowed.to_owned());
                    }
                } else {
                    list_items[self.items.len() - 1].item = Some(overflowed.to_owned());
                }
            }
            (true, DefinitiveListTactic::Horizontal, placeholder @ Some(..)) => {
                list_items[self.items.len() - 1].item = placeholder;
            }
            _ if !self.items.is_empty() => {
                list_items[self.items.len() - 1].item = self
                    .items
                    .last()
                    .and_then(|last_item| last_item.rewrite(self.context, self.nested_shape));

                // Use horizontal layout for a function with a single argument as long as
                // everything fits in a single line.
                // `self.one_line_width == 0` means vertical layout is forced.
                if self.items.len() == 1
                    && self.one_line_width != 0
                    && !list_items[0].has_comment()
                    && !list_items[0].inner_as_ref().contains('\n')
                    && crate::lists::total_item_width(&list_items[0]) <= self.one_line_width
                {
                    tactic = DefinitiveListTactic::Horizontal;
                } else {
                    tactic = self.default_tactic(list_items);

                    if tactic == DefinitiveListTactic::Vertical {
                        if let Some((all_simple, num_args_before)) =
                            maybe_get_args_offset(self.ident, &self.items)
                        {
                            let one_line = all_simple
                                && definitive_tactic(
                                    &list_items[..num_args_before],
                                    ListTactic::HorizontalVertical,
                                    Separator::Comma,
                                    self.nested_shape.width,
                                ) == DefinitiveListTactic::Horizontal
                                && definitive_tactic(
                                    &list_items[num_args_before + 1..],
                                    ListTactic::HorizontalVertical,
                                    Separator::Comma,
                                    self.nested_shape.width,
                                ) == DefinitiveListTactic::Horizontal;

                            if one_line {
                                tactic = DefinitiveListTactic::SpecialMacro(num_args_before);
                            };
                        } else if is_every_expr_simple(&self.items) && no_long_items(list_items) {
                            tactic = DefinitiveListTactic::Mixed;
                        }
                    }
                }
            }
            _ => (),
        }

        tactic
    }

    fn rewrite_items(&self) -> Option<(bool, String)> {
        let span = self.items_span();
        let items = itemize_list(
            self.context.snippet_provider,
            self.items.iter(),
            self.suffix,
            ",",
            |item| item.span().lo(),
            |item| item.span().hi(),
            |item| item.rewrite(self.context, self.nested_shape),
            span.lo(),
            span.hi(),
            true,
        );
        let mut list_items: Vec<_> = items.collect();

        // Try letting the last argument overflow to the next line with block
        // indentation. If its first line fits on one line with the other arguments,
        // we format the function arguments horizontally.
        let tactic = self.try_overflow_last_item(&mut list_items);
        let trailing_separator = if let Some(tactic) = self.force_separator_tactic {
            tactic
        } else if !self.context.use_block_indent() {
            SeparatorTactic::Never
        } else {
            self.context.config.trailing_comma()
        };
        let ends_with_newline = match tactic {
            DefinitiveListTactic::Vertical | DefinitiveListTactic::Mixed => {
                self.context.use_block_indent()
            }
            _ => false,
        };

        let fmt = ListFormatting::new(self.nested_shape, self.context.config)
            .tactic(tactic)
            .trailing_separator(trailing_separator)
            .ends_with_newline(ends_with_newline);

        write_list(&list_items, &fmt)
            .map(|items_str| (tactic == DefinitiveListTactic::Horizontal, items_str))
    }

    fn wrap_items(&self, items_str: &str, shape: Shape, is_extendable: bool) -> String {
        let shape = Shape {
            width: shape.width.saturating_sub(last_line_width(self.ident)),
            ..shape
        };

        let (prefix, suffix) = match self.custom_delims {
            Some((lhs, rhs)) => (lhs, rhs),
            _ => (self.prefix, self.suffix),
        };

        let extend_width = if items_str.is_empty() {
            2
        } else {
            first_line_width(items_str) + 1
        };
        let nested_indent_str = self
            .nested_shape
            .indent
            .to_string_with_newline(self.context.config);
        let indent_str = shape
            .block()
            .indent
            .to_string_with_newline(self.context.config);
        let mut result = String::with_capacity(
            self.ident.len() + items_str.len() + 2 + indent_str.len() + nested_indent_str.len(),
        );
        result.push_str(self.ident);
        result.push_str(prefix);
        let force_single_line = if self.context.config.version() == Version::Two {
            !self.context.use_block_indent() || (is_extendable && extend_width <= shape.width)
        } else {
            // 2 = `()`
            let fits_one_line = items_str.len() + 2 <= shape.width;
            !self.context.use_block_indent()
                || (self.context.inside_macro() && !items_str.contains('\n') && fits_one_line)
                || (is_extendable && extend_width <= shape.width)
        };
        if force_single_line {
            result.push_str(items_str);
        } else {
            if !items_str.is_empty() {
                result.push_str(&nested_indent_str);
                result.push_str(items_str);
            }
            result.push_str(&indent_str);
        }
        result.push_str(suffix);
        result
    }

    fn rewrite(&self, shape: Shape) -> Option<String> {
        let (extendable, items_str) = self.rewrite_items()?;

        // If we are using visual indent style and failed to format, retry with block indent.
        if !self.context.use_block_indent()
            && need_block_indent(&items_str, self.nested_shape)
            && !extendable
        {
            self.context.use_block.replace(true);
            let result = self.rewrite(shape);
            self.context.use_block.replace(false);
            return result;
        }

        Some(self.wrap_items(&items_str, shape, extendable))
    }
}

fn need_block_indent(s: &str, shape: Shape) -> bool {
    s.lines().skip(1).any(|s| {
        s.find(|c| !char::is_whitespace(c))
            .map_or(false, |w| w + 1 < shape.indent.width())
    })
}

fn can_be_overflowed(context: &RewriteContext<'_>, items: &[OverflowableItem<'_>]) -> bool {
    items
        .last()
        .map_or(false, |x| x.can_be_overflowed(context, items.len()))
}

/// Returns a shape for the last argument which is going to be overflowed.
fn last_item_shape(
    lists: &[OverflowableItem<'_>],
    items: &[ListItem],
    shape: Shape,
    args_max_width: usize,
) -> Option<Shape> {
    if items.len() == 1 && !lists.get(0)?.is_nested_call() {
        return Some(shape);
    }
    let offset = items
        .iter()
        .dropping_back(1)
        .map(|i| {
            // 2 = ", "
            2 + i.inner_as_ref().len()
        })
        .sum();
    Shape {
        width: min(args_max_width, shape.width),
        ..shape
    }
    .offset_left(offset)
}

fn shape_from_indent_style(
    context: &RewriteContext<'_>,
    shape: Shape,
    overhead: usize,
    offset: usize,
) -> Shape {
    let (shape, overhead) = if context.use_block_indent() {
        let shape = shape
            .block()
            .block_indent(context.config.tab_spaces())
            .with_max_width(context.config);
        (shape, 1) // 1 = ","
    } else {
        (shape.visual_indent(offset), overhead)
    };
    Shape {
        width: shape.width.saturating_sub(overhead),
        ..shape
    }
}

fn no_long_items(list: &[ListItem]) -> bool {
    list.iter()
        .all(|item| item.inner_as_ref().len() <= SHORT_ITEM_THRESHOLD)
}

/// In case special-case style is required, returns an offset from which we start horizontal layout.
pub(crate) fn maybe_get_args_offset(
    callee_str: &str,
    args: &[OverflowableItem<'_>],
) -> Option<(bool, usize)> {
    if let Some(&(_, num_args_before)) = args
        .get(0)?
        .whitelist()
        .iter()
        .find(|&&(s, _)| s == callee_str)
    {
        let all_simple = args.len() > num_args_before
            && is_every_expr_simple(&args[0..num_args_before])
            && is_every_expr_simple(&args[num_args_before + 1..]);

        Some((all_simple, num_args_before))
    } else {
        None
    }
}
