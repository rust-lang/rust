use super::{Parser, PResult, PathStyle, SemiColonMode, BlockMode};

use crate::maybe_whole;
use crate::ptr::P;
use crate::ast::{self, Ident, Attribute, AttrStyle};
use crate::ast::{Item, ItemKind, ImplItem, TraitItem, TraitItemKind};
use crate::ast::{UseTree, UseTreeKind, PathSegment};
use crate::ast::{IsAuto, Constness, IsAsync, Unsafety, Defaultness};
use crate::ast::{Visibility, VisibilityKind, Mutability, FnDecl, FnHeader};
use crate::ast::{ForeignItem, ForeignItemKind};
use crate::ast::{Ty, TyKind, GenericBounds, TraitRef};
use crate::ast::{EnumDef, VariantData, StructField, AnonConst};
use crate::ast::{Mac, MacDelimiter};
use crate::ext::base::DummyResult;
use crate::parse::token;
use crate::parse::parser::maybe_append;
use crate::parse::diagnostics::{Error};
use crate::tokenstream::{TokenTree, TokenStream};
use crate::source_map::{respan, Span, Spanned};
use crate::symbol::{kw, sym};

use std::mem;
use log::debug;
use rustc_target::spec::abi::{Abi};
use errors::{Applicability, DiagnosticBuilder, DiagnosticId};

/// Whether the type alias or associated type is a concrete type or an opaque type
#[derive(Debug)]
pub enum AliasKind {
    /// Just a new name for the same type
    Weak(P<Ty>),
    /// Only trait impls of the type will be usable, not the actual type itself
    OpaqueTy(GenericBounds),
}

pub(super) type ItemInfo = (Ident, ItemKind, Option<Vec<Attribute>>);

impl<'a> Parser<'a> {
    pub fn parse_item(&mut self) -> PResult<'a, Option<P<Item>>> {
        let attrs = self.parse_outer_attributes()?;
        self.parse_item_(attrs, true, false)
    }

    pub(super) fn parse_item_(
        &mut self,
        attrs: Vec<Attribute>,
        macros_allowed: bool,
        attributes_allowed: bool,
    ) -> PResult<'a, Option<P<Item>>> {
        let mut unclosed_delims = vec![];
        let (ret, tokens) = self.collect_tokens(|this| {
            let item = this.parse_item_implementation(attrs, macros_allowed, attributes_allowed);
            unclosed_delims.append(&mut this.unclosed_delims);
            item
        })?;
        self.unclosed_delims.append(&mut unclosed_delims);

        // Once we've parsed an item and recorded the tokens we got while
        // parsing we may want to store `tokens` into the item we're about to
        // return. Note, though, that we specifically didn't capture tokens
        // related to outer attributes. The `tokens` field here may later be
        // used with procedural macros to convert this item back into a token
        // stream, but during expansion we may be removing attributes as we go
        // along.
        //
        // If we've got inner attributes then the `tokens` we've got above holds
        // these inner attributes. If an inner attribute is expanded we won't
        // actually remove it from the token stream, so we'll just keep yielding
        // it (bad!). To work around this case for now we just avoid recording
        // `tokens` if we detect any inner attributes. This should help keep
        // expansion correct, but we should fix this bug one day!
        Ok(ret.map(|item| {
            item.map(|mut i| {
                if !i.attrs.iter().any(|attr| attr.style == AttrStyle::Inner) {
                    i.tokens = Some(tokens);
                }
                i
            })
        }))
    }

    /// Parses one of the items allowed by the flags.
    fn parse_item_implementation(
        &mut self,
        attrs: Vec<Attribute>,
        macros_allowed: bool,
        attributes_allowed: bool,
    ) -> PResult<'a, Option<P<Item>>> {
        maybe_whole!(self, NtItem, |item| {
            let mut item = item.into_inner();
            let mut attrs = attrs;
            mem::swap(&mut item.attrs, &mut attrs);
            item.attrs.extend(attrs);
            Some(P(item))
        });

        let lo = self.token.span;

        let visibility = self.parse_visibility(false)?;

        if self.eat_keyword(kw::Use) {
            // USE ITEM
            let item_ = ItemKind::Use(P(self.parse_use_tree()?));
            self.expect(&token::Semi)?;

            let span = lo.to(self.prev_span);
            let item =
                self.mk_item(span, Ident::invalid(), item_, visibility, attrs);
            return Ok(Some(item));
        }

        if self.eat_keyword(kw::Extern) {
            let extern_sp = self.prev_span;
            if self.eat_keyword(kw::Crate) {
                return Ok(Some(self.parse_item_extern_crate(lo, visibility, attrs)?));
            }

            let opt_abi = self.parse_opt_abi()?;

            if self.eat_keyword(kw::Fn) {
                // EXTERN FUNCTION ITEM
                let fn_span = self.prev_span;
                let abi = opt_abi.unwrap_or(Abi::C);
                let (ident, item_, extra_attrs) =
                    self.parse_item_fn(Unsafety::Normal,
                                       respan(fn_span, IsAsync::NotAsync),
                                       respan(fn_span, Constness::NotConst),
                                       abi)?;
                let prev_span = self.prev_span;
                let item = self.mk_item(lo.to(prev_span),
                                        ident,
                                        item_,
                                        visibility,
                                        maybe_append(attrs, extra_attrs));
                return Ok(Some(item));
            } else if self.check(&token::OpenDelim(token::Brace)) {
                return Ok(Some(
                    self.parse_item_foreign_mod(lo, opt_abi, visibility, attrs, extern_sp)?,
                ));
            }

            self.unexpected()?;
        }

        if self.is_static_global() {
            self.bump();
            // STATIC ITEM
            let m = self.parse_mutability();
            let (ident, item_, extra_attrs) = self.parse_item_const(Some(m))?;
            let prev_span = self.prev_span;
            let item = self.mk_item(lo.to(prev_span),
                                    ident,
                                    item_,
                                    visibility,
                                    maybe_append(attrs, extra_attrs));
            return Ok(Some(item));
        }
        if self.eat_keyword(kw::Const) {
            let const_span = self.prev_span;
            if self.check_keyword(kw::Fn)
                || (self.check_keyword(kw::Unsafe)
                    && self.is_keyword_ahead(1, &[kw::Fn])) {
                // CONST FUNCTION ITEM
                let unsafety = self.parse_unsafety();
                self.bump();
                let (ident, item_, extra_attrs) =
                    self.parse_item_fn(unsafety,
                                       respan(const_span, IsAsync::NotAsync),
                                       respan(const_span, Constness::Const),
                                       Abi::Rust)?;
                let prev_span = self.prev_span;
                let item = self.mk_item(lo.to(prev_span),
                                        ident,
                                        item_,
                                        visibility,
                                        maybe_append(attrs, extra_attrs));
                return Ok(Some(item));
            }

            // CONST ITEM
            if self.eat_keyword(kw::Mut) {
                let prev_span = self.prev_span;
                self.struct_span_err(prev_span, "const globals cannot be mutable")
                    .span_label(prev_span, "cannot be mutable")
                    .span_suggestion(
                        const_span,
                        "you might want to declare a static instead",
                        "static".to_owned(),
                        Applicability::MaybeIncorrect,
                    )
                    .emit();
            }
            let (ident, item_, extra_attrs) = self.parse_item_const(None)?;
            let prev_span = self.prev_span;
            let item = self.mk_item(lo.to(prev_span),
                                    ident,
                                    item_,
                                    visibility,
                                    maybe_append(attrs, extra_attrs));
            return Ok(Some(item));
        }

        // Parse `async unsafe? fn`.
        if self.check_keyword(kw::Async) {
            let async_span = self.token.span;
            if self.is_keyword_ahead(1, &[kw::Fn])
                || self.is_keyword_ahead(2, &[kw::Fn])
            {
                // ASYNC FUNCTION ITEM
                self.bump(); // `async`
                let unsafety = self.parse_unsafety(); // `unsafe`?
                self.expect_keyword(kw::Fn)?; // `fn`
                let fn_span = self.prev_span;
                let (ident, item_, extra_attrs) =
                    self.parse_item_fn(unsafety,
                                    respan(async_span, IsAsync::Async {
                                        closure_id: ast::DUMMY_NODE_ID,
                                        return_impl_trait_id: ast::DUMMY_NODE_ID,
                                    }),
                                    respan(fn_span, Constness::NotConst),
                                    Abi::Rust)?;
                let prev_span = self.prev_span;
                let item = self.mk_item(lo.to(prev_span),
                                        ident,
                                        item_,
                                        visibility,
                                        maybe_append(attrs, extra_attrs));
                self.ban_async_in_2015(async_span);
                return Ok(Some(item));
            }
        }
        if self.check_keyword(kw::Unsafe) &&
            self.is_keyword_ahead(1, &[kw::Trait, kw::Auto])
        {
            // UNSAFE TRAIT ITEM
            self.bump(); // `unsafe`
            let is_auto = if self.eat_keyword(kw::Trait) {
                IsAuto::No
            } else {
                self.expect_keyword(kw::Auto)?;
                self.expect_keyword(kw::Trait)?;
                IsAuto::Yes
            };
            let (ident, item_, extra_attrs) =
                self.parse_item_trait(is_auto, Unsafety::Unsafe)?;
            let prev_span = self.prev_span;
            let item = self.mk_item(lo.to(prev_span),
                                    ident,
                                    item_,
                                    visibility,
                                    maybe_append(attrs, extra_attrs));
            return Ok(Some(item));
        }
        if self.check_keyword(kw::Impl) ||
           self.check_keyword(kw::Unsafe) &&
                self.is_keyword_ahead(1, &[kw::Impl]) ||
           self.check_keyword(kw::Default) &&
                self.is_keyword_ahead(1, &[kw::Impl, kw::Unsafe]) {
            // IMPL ITEM
            let defaultness = self.parse_defaultness();
            let unsafety = self.parse_unsafety();
            self.expect_keyword(kw::Impl)?;
            let (ident, item, extra_attrs) = self.parse_item_impl(unsafety, defaultness)?;
            let span = lo.to(self.prev_span);
            return Ok(Some(self.mk_item(span, ident, item, visibility,
                                        maybe_append(attrs, extra_attrs))));
        }
        if self.check_keyword(kw::Fn) {
            // FUNCTION ITEM
            self.bump();
            let fn_span = self.prev_span;
            let (ident, item_, extra_attrs) =
                self.parse_item_fn(Unsafety::Normal,
                                   respan(fn_span, IsAsync::NotAsync),
                                   respan(fn_span, Constness::NotConst),
                                   Abi::Rust)?;
            let prev_span = self.prev_span;
            let item = self.mk_item(lo.to(prev_span),
                                    ident,
                                    item_,
                                    visibility,
                                    maybe_append(attrs, extra_attrs));
            return Ok(Some(item));
        }
        if self.check_keyword(kw::Unsafe)
            && self.look_ahead(1, |t| *t != token::OpenDelim(token::Brace)) {
            // UNSAFE FUNCTION ITEM
            self.bump(); // `unsafe`
            // `{` is also expected after `unsafe`, in case of error, include it in the diagnostic
            self.check(&token::OpenDelim(token::Brace));
            let abi = if self.eat_keyword(kw::Extern) {
                self.parse_opt_abi()?.unwrap_or(Abi::C)
            } else {
                Abi::Rust
            };
            self.expect_keyword(kw::Fn)?;
            let fn_span = self.prev_span;
            let (ident, item_, extra_attrs) =
                self.parse_item_fn(Unsafety::Unsafe,
                                   respan(fn_span, IsAsync::NotAsync),
                                   respan(fn_span, Constness::NotConst),
                                   abi)?;
            let prev_span = self.prev_span;
            let item = self.mk_item(lo.to(prev_span),
                                    ident,
                                    item_,
                                    visibility,
                                    maybe_append(attrs, extra_attrs));
            return Ok(Some(item));
        }
        if self.eat_keyword(kw::Mod) {
            // MODULE ITEM
            let (ident, item_, extra_attrs) =
                self.parse_item_mod(&attrs[..])?;
            let prev_span = self.prev_span;
            let item = self.mk_item(lo.to(prev_span),
                                    ident,
                                    item_,
                                    visibility,
                                    maybe_append(attrs, extra_attrs));
            return Ok(Some(item));
        }
        if let Some(type_) = self.eat_type() {
            let (ident, alias, generics) = type_?;
            // TYPE ITEM
            let item_ = match alias {
                AliasKind::Weak(ty) => ItemKind::TyAlias(ty, generics),
                AliasKind::OpaqueTy(bounds) => ItemKind::OpaqueTy(bounds, generics),
            };
            let prev_span = self.prev_span;
            let item = self.mk_item(lo.to(prev_span),
                                    ident,
                                    item_,
                                    visibility,
                                    attrs);
            return Ok(Some(item));
        }
        if self.eat_keyword(kw::Enum) {
            // ENUM ITEM
            let (ident, item_, extra_attrs) = self.parse_item_enum()?;
            let prev_span = self.prev_span;
            let item = self.mk_item(lo.to(prev_span),
                                    ident,
                                    item_,
                                    visibility,
                                    maybe_append(attrs, extra_attrs));
            return Ok(Some(item));
        }
        if self.check_keyword(kw::Trait)
            || (self.check_keyword(kw::Auto)
                && self.is_keyword_ahead(1, &[kw::Trait]))
        {
            let is_auto = if self.eat_keyword(kw::Trait) {
                IsAuto::No
            } else {
                self.expect_keyword(kw::Auto)?;
                self.expect_keyword(kw::Trait)?;
                IsAuto::Yes
            };
            // TRAIT ITEM
            let (ident, item_, extra_attrs) =
                self.parse_item_trait(is_auto, Unsafety::Normal)?;
            let prev_span = self.prev_span;
            let item = self.mk_item(lo.to(prev_span),
                                    ident,
                                    item_,
                                    visibility,
                                    maybe_append(attrs, extra_attrs));
            return Ok(Some(item));
        }
        if self.eat_keyword(kw::Struct) {
            // STRUCT ITEM
            let (ident, item_, extra_attrs) = self.parse_item_struct()?;
            let prev_span = self.prev_span;
            let item = self.mk_item(lo.to(prev_span),
                                    ident,
                                    item_,
                                    visibility,
                                    maybe_append(attrs, extra_attrs));
            return Ok(Some(item));
        }
        if self.is_union_item() {
            // UNION ITEM
            self.bump();
            let (ident, item_, extra_attrs) = self.parse_item_union()?;
            let prev_span = self.prev_span;
            let item = self.mk_item(lo.to(prev_span),
                                    ident,
                                    item_,
                                    visibility,
                                    maybe_append(attrs, extra_attrs));
            return Ok(Some(item));
        }
        if let Some(macro_def) = self.eat_macro_def(&attrs, &visibility, lo)? {
            return Ok(Some(macro_def));
        }

        // Verify whether we have encountered a struct or method definition where the user forgot to
        // add the `struct` or `fn` keyword after writing `pub`: `pub S {}`
        if visibility.node.is_pub() &&
            self.check_ident() &&
            self.look_ahead(1, |t| *t != token::Not)
        {
            // Space between `pub` keyword and the identifier
            //
            //     pub   S {}
            //        ^^^ `sp` points here
            let sp = self.prev_span.between(self.token.span);
            let full_sp = self.prev_span.to(self.token.span);
            let ident_sp = self.token.span;
            if self.look_ahead(1, |t| *t == token::OpenDelim(token::Brace)) {
                // possible public struct definition where `struct` was forgotten
                let ident = self.parse_ident().unwrap();
                let msg = format!("add `struct` here to parse `{}` as a public struct",
                                  ident);
                let mut err = self.diagnostic()
                    .struct_span_err(sp, "missing `struct` for struct definition");
                err.span_suggestion_short(
                    sp, &msg, " struct ".into(), Applicability::MaybeIncorrect // speculative
                );
                return Err(err);
            } else if self.look_ahead(1, |t| *t == token::OpenDelim(token::Paren)) {
                let ident = self.parse_ident().unwrap();
                self.bump();  // `(`
                let kw_name = if let Ok(Some(_)) = self.parse_self_parameter_with_attrs()
                    .map_err(|mut e| e.cancel())
                {
                    "method"
                } else {
                    "function"
                };
                self.consume_block(token::Paren);
                let (kw, kw_name, ambiguous) = if self.check(&token::RArrow) {
                    self.eat_to_tokens(&[&token::OpenDelim(token::Brace)]);
                    self.bump();  // `{`
                    ("fn", kw_name, false)
                } else if self.check(&token::OpenDelim(token::Brace)) {
                    self.bump();  // `{`
                    ("fn", kw_name, false)
                } else if self.check(&token::Colon) {
                    let kw = "struct";
                    (kw, kw, false)
                } else {
                    ("fn` or `struct", "function or struct", true)
                };

                let msg = format!("missing `{}` for {} definition", kw, kw_name);
                let mut err = self.diagnostic().struct_span_err(sp, &msg);
                if !ambiguous {
                    self.consume_block(token::Brace);
                    let suggestion = format!("add `{}` here to parse `{}` as a public {}",
                                             kw,
                                             ident,
                                             kw_name);
                    err.span_suggestion_short(
                        sp, &suggestion, format!(" {} ", kw), Applicability::MachineApplicable
                    );
                } else {
                    if let Ok(snippet) = self.span_to_snippet(ident_sp) {
                        err.span_suggestion(
                            full_sp,
                            "if you meant to call a macro, try",
                            format!("{}!", snippet),
                            // this is the `ambiguous` conditional branch
                            Applicability::MaybeIncorrect
                        );
                    } else {
                        err.help("if you meant to call a macro, remove the `pub` \
                                  and add a trailing `!` after the identifier");
                    }
                }
                return Err(err);
            } else if self.look_ahead(1, |t| *t == token::Lt) {
                let ident = self.parse_ident().unwrap();
                self.eat_to_tokens(&[&token::Gt]);
                self.bump();  // `>`
                let (kw, kw_name, ambiguous) = if self.eat(&token::OpenDelim(token::Paren)) {
                    if let Ok(Some(_)) = self.parse_self_parameter_with_attrs()
                        .map_err(|mut e| e.cancel())
                    {
                        ("fn", "method", false)
                    } else {
                        ("fn", "function", false)
                    }
                } else if self.check(&token::OpenDelim(token::Brace)) {
                    ("struct", "struct", false)
                } else {
                    ("fn` or `struct", "function or struct", true)
                };
                let msg = format!("missing `{}` for {} definition", kw, kw_name);
                let mut err = self.diagnostic().struct_span_err(sp, &msg);
                if !ambiguous {
                    err.span_suggestion_short(
                        sp,
                        &format!("add `{}` here to parse `{}` as a public {}", kw, ident, kw_name),
                        format!(" {} ", kw),
                        Applicability::MachineApplicable,
                    );
                }
                return Err(err);
            }
        }
        self.parse_macro_use_or_failure(attrs, macros_allowed, attributes_allowed, lo, visibility)
    }

    /// This is the fall-through for parsing items.
    fn parse_macro_use_or_failure(
        &mut self,
        attrs: Vec<Attribute> ,
        macros_allowed: bool,
        attributes_allowed: bool,
        lo: Span,
        visibility: Visibility
    ) -> PResult<'a, Option<P<Item>>> {
        if macros_allowed && self.token.is_path_start() &&
                !(self.is_async_fn() && self.token.span.rust_2015()) {
            // MACRO INVOCATION ITEM

            let prev_span = self.prev_span;
            self.complain_if_pub_macro(&visibility.node, prev_span);

            let mac_lo = self.token.span;

            // item macro.
            let path = self.parse_path(PathStyle::Mod)?;
            self.expect(&token::Not)?;
            let (delim, tts) = self.expect_delimited_token_tree()?;
            if delim != MacDelimiter::Brace && !self.eat(&token::Semi) {
                self.report_invalid_macro_expansion_item();
            }

            let hi = self.prev_span;
            let mac = Mac {
                path,
                tts,
                delim,
                span: mac_lo.to(hi),
                prior_type_ascription: self.last_type_ascription,
            };
            let item =
                self.mk_item(lo.to(hi), Ident::invalid(), ItemKind::Mac(mac), visibility, attrs);
            return Ok(Some(item));
        }

        // FAILURE TO PARSE ITEM
        match visibility.node {
            VisibilityKind::Inherited => {}
            _ => {
                return Err(self.span_fatal(self.prev_span, "unmatched visibility `pub`"));
            }
        }

        if !attributes_allowed && !attrs.is_empty() {
            self.expected_item_err(&attrs)?;
        }
        Ok(None)
    }

    /// Emits an expected-item-after-attributes error.
    fn expected_item_err(&mut self, attrs: &[Attribute]) -> PResult<'a,  ()> {
        let message = match attrs.last() {
            Some(&Attribute { is_sugared_doc: true, .. }) => "expected item after doc comment",
            _ => "expected item after attributes",
        };

        let mut err = self.diagnostic().struct_span_err(self.prev_span, message);
        if attrs.last().unwrap().is_sugared_doc {
            err.span_label(self.prev_span, "this doc comment doesn't document anything");
        }
        Err(err)
    }

    pub(super) fn is_async_fn(&self) -> bool {
        self.token.is_keyword(kw::Async) &&
            self.is_keyword_ahead(1, &[kw::Fn])
    }

    /// Parses a macro invocation inside a `trait`, `impl` or `extern` block.
    fn parse_assoc_macro_invoc(&mut self, item_kind: &str, vis: Option<&Visibility>,
                               at_end: &mut bool) -> PResult<'a, Option<Mac>>
    {
        if self.token.is_path_start() &&
                !(self.is_async_fn() && self.token.span.rust_2015()) {
            let prev_span = self.prev_span;
            let lo = self.token.span;
            let path = self.parse_path(PathStyle::Mod)?;

            if path.segments.len() == 1 {
                if !self.eat(&token::Not) {
                    return Err(self.missing_assoc_item_kind_err(item_kind, prev_span));
                }
            } else {
                self.expect(&token::Not)?;
            }

            if let Some(vis) = vis {
                self.complain_if_pub_macro(&vis.node, prev_span);
            }

            *at_end = true;

            // eat a matched-delimiter token tree:
            let (delim, tts) = self.expect_delimited_token_tree()?;
            if delim != MacDelimiter::Brace {
                self.expect(&token::Semi)?;
            }

            Ok(Some(Mac {
                path,
                tts,
                delim,
                span: lo.to(self.prev_span),
                prior_type_ascription: self.last_type_ascription,
            }))
        } else {
            Ok(None)
        }
    }

    fn missing_assoc_item_kind_err(&self, item_type: &str, prev_span: Span)
                                   -> DiagnosticBuilder<'a>
    {
        let expected_kinds = if item_type == "extern" {
            "missing `fn`, `type`, or `static`"
        } else {
            "missing `fn`, `type`, or `const`"
        };

        // Given this code `path(`, it seems like this is not
        // setting the visibility of a macro invocation, but rather
        // a mistyped method declaration.
        // Create a diagnostic pointing out that `fn` is missing.
        //
        // x |     pub path(&self) {
        //   |        ^ missing `fn`, `type`, or `const`
        //     pub  path(
        //        ^^ `sp` below will point to this
        let sp = prev_span.between(self.prev_span);
        let mut err = self.diagnostic().struct_span_err(
            sp,
            &format!("{} for {}-item declaration",
                     expected_kinds, item_type));
        err.span_label(sp, expected_kinds);
        err
    }

    /// Parses an implementation item, `impl` keyword is already parsed.
    ///
    ///    impl<'a, T> TYPE { /* impl items */ }
    ///    impl<'a, T> TRAIT for TYPE { /* impl items */ }
    ///    impl<'a, T> !TRAIT for TYPE { /* impl items */ }
    ///
    /// We actually parse slightly more relaxed grammar for better error reporting and recovery.
    ///     `impl` GENERICS `!`? TYPE `for`? (TYPE | `..`) (`where` PREDICATES)? `{` BODY `}`
    ///     `impl` GENERICS `!`? TYPE (`where` PREDICATES)? `{` BODY `}`
    fn parse_item_impl(&mut self, unsafety: Unsafety, defaultness: Defaultness)
                       -> PResult<'a, ItemInfo> {
        // First, parse generic parameters if necessary.
        let mut generics = if self.choose_generics_over_qpath() {
            self.parse_generics()?
        } else {
            ast::Generics::default()
        };

        // Disambiguate `impl !Trait for Type { ... }` and `impl ! { ... }` for the never type.
        let polarity = if self.check(&token::Not) && self.look_ahead(1, |t| t.can_begin_type()) {
            self.bump(); // `!`
            ast::ImplPolarity::Negative
        } else {
            ast::ImplPolarity::Positive
        };

        // Parse both types and traits as a type, then reinterpret if necessary.
        let err_path = |span| ast::Path::from_ident(Ident::new(kw::Invalid, span));
        let ty_first = if self.token.is_keyword(kw::For) &&
                          self.look_ahead(1, |t| t != &token::Lt) {
            let span = self.prev_span.between(self.token.span);
            self.struct_span_err(span, "missing trait in a trait impl").emit();
            P(Ty { node: TyKind::Path(None, err_path(span)), span, id: ast::DUMMY_NODE_ID })
        } else {
            self.parse_ty()?
        };

        // If `for` is missing we try to recover.
        let has_for = self.eat_keyword(kw::For);
        let missing_for_span = self.prev_span.between(self.token.span);

        let ty_second = if self.token == token::DotDot {
            // We need to report this error after `cfg` expansion for compatibility reasons
            self.bump(); // `..`, do not add it to expected tokens
            Some(DummyResult::raw_ty(self.prev_span, true))
        } else if has_for || self.token.can_begin_type() {
            Some(self.parse_ty()?)
        } else {
            None
        };

        generics.where_clause = self.parse_where_clause()?;

        let (impl_items, attrs) = self.parse_impl_body()?;

        let item_kind = match ty_second {
            Some(ty_second) => {
                // impl Trait for Type
                if !has_for {
                    self.struct_span_err(missing_for_span, "missing `for` in a trait impl")
                        .span_suggestion_short(
                            missing_for_span,
                            "add `for` here",
                            " for ".to_string(),
                            Applicability::MachineApplicable,
                        ).emit();
                }

                let ty_first = ty_first.into_inner();
                let path = match ty_first.node {
                    // This notably includes paths passed through `ty` macro fragments (#46438).
                    TyKind::Path(None, path) => path,
                    _ => {
                        self.span_err(ty_first.span, "expected a trait, found type");
                        err_path(ty_first.span)
                    }
                };
                let trait_ref = TraitRef { path, ref_id: ty_first.id };

                ItemKind::Impl(unsafety, polarity, defaultness,
                               generics, Some(trait_ref), ty_second, impl_items)
            }
            None => {
                // impl Type
                ItemKind::Impl(unsafety, polarity, defaultness,
                               generics, None, ty_first, impl_items)
            }
        };

        Ok((Ident::invalid(), item_kind, Some(attrs)))
    }

    fn parse_impl_body(&mut self) -> PResult<'a, (Vec<ImplItem>, Vec<Attribute>)> {
        self.expect(&token::OpenDelim(token::Brace))?;
        let attrs = self.parse_inner_attributes()?;

        let mut impl_items = Vec::new();
        while !self.eat(&token::CloseDelim(token::Brace)) {
            let mut at_end = false;
            match self.parse_impl_item(&mut at_end) {
                Ok(impl_item) => impl_items.push(impl_item),
                Err(mut err) => {
                    err.emit();
                    if !at_end {
                        self.recover_stmt_(SemiColonMode::Break, BlockMode::Break);
                    }
                }
            }
        }
        Ok((impl_items, attrs))
    }

    /// Parses an impl item.
    pub fn parse_impl_item(&mut self, at_end: &mut bool) -> PResult<'a, ImplItem> {
        maybe_whole!(self, NtImplItem, |x| x);
        let attrs = self.parse_outer_attributes()?;
        let mut unclosed_delims = vec![];
        let (mut item, tokens) = self.collect_tokens(|this| {
            let item = this.parse_impl_item_(at_end, attrs);
            unclosed_delims.append(&mut this.unclosed_delims);
            item
        })?;
        self.unclosed_delims.append(&mut unclosed_delims);

        // See `parse_item` for why this clause is here.
        if !item.attrs.iter().any(|attr| attr.style == AttrStyle::Inner) {
            item.tokens = Some(tokens);
        }
        Ok(item)
    }

    fn parse_impl_item_(&mut self,
                        at_end: &mut bool,
                        mut attrs: Vec<Attribute>) -> PResult<'a, ImplItem> {
        let lo = self.token.span;
        let vis = self.parse_visibility(false)?;
        let defaultness = self.parse_defaultness();
        let (name, node, generics) = if let Some(type_) = self.eat_type() {
            let (name, alias, generics) = type_?;
            let kind = match alias {
                AliasKind::Weak(typ) => ast::ImplItemKind::TyAlias(typ),
                AliasKind::OpaqueTy(bounds) => ast::ImplItemKind::OpaqueTy(bounds),
            };
            (name, kind, generics)
        } else if self.is_const_item() {
            // This parses the grammar:
            //     ImplItemConst = "const" Ident ":" Ty "=" Expr ";"
            self.expect_keyword(kw::Const)?;
            let name = self.parse_ident()?;
            self.expect(&token::Colon)?;
            let typ = self.parse_ty()?;
            self.expect(&token::Eq)?;
            let expr = self.parse_expr()?;
            self.expect(&token::Semi)?;
            (name, ast::ImplItemKind::Const(typ, expr), ast::Generics::default())
        } else {
            let (name, inner_attrs, generics, node) = self.parse_impl_method(&vis, at_end)?;
            attrs.extend(inner_attrs);
            (name, node, generics)
        };

        Ok(ImplItem {
            id: ast::DUMMY_NODE_ID,
            span: lo.to(self.prev_span),
            ident: name,
            vis,
            defaultness,
            attrs,
            generics,
            node,
            tokens: None,
        })
    }

    /// Parses defaultness (i.e., `default` or nothing).
    fn parse_defaultness(&mut self) -> Defaultness {
        // `pub` is included for better error messages
        if self.check_keyword(kw::Default) &&
            self.is_keyword_ahead(1, &[
                kw::Impl,
                kw::Const,
                kw::Async,
                kw::Fn,
                kw::Unsafe,
                kw::Extern,
                kw::Type,
                kw::Pub,
            ])
        {
            self.bump(); // `default`
            Defaultness::Default
        } else {
            Defaultness::Final
        }
    }

    /// Returns `true` if we are looking at `const ID`
    /// (returns `false` for things like `const fn`, etc.).
    fn is_const_item(&self) -> bool {
        self.token.is_keyword(kw::Const) &&
            !self.is_keyword_ahead(1, &[kw::Fn, kw::Unsafe])
    }

    /// Parse a method or a macro invocation in a trait impl.
    fn parse_impl_method(&mut self, vis: &Visibility, at_end: &mut bool)
                         -> PResult<'a, (Ident, Vec<Attribute>, ast::Generics,
                             ast::ImplItemKind)> {
        // code copied from parse_macro_use_or_failure... abstraction!
        if let Some(mac) = self.parse_assoc_macro_invoc("impl", Some(vis), at_end)? {
            // method macro
            Ok((Ident::invalid(), vec![], ast::Generics::default(),
                ast::ImplItemKind::Macro(mac)))
        } else {
            let (constness, unsafety, asyncness, abi) = self.parse_fn_front_matter()?;
            let ident = self.parse_ident()?;
            let mut generics = self.parse_generics()?;
            let decl = self.parse_fn_decl_with_self(|p| {
                p.parse_param_general(true, false, |_| true)
            })?;
            generics.where_clause = self.parse_where_clause()?;
            *at_end = true;
            let (inner_attrs, body) = self.parse_inner_attrs_and_block()?;
            let header = ast::FnHeader { abi, unsafety, constness, asyncness };
            Ok((ident, inner_attrs, generics, ast::ImplItemKind::Method(
                ast::MethodSig { header, decl },
                body
            )))
        }
    }

    /// Parses all the "front matter" for a `fn` declaration, up to
    /// and including the `fn` keyword:
    ///
    /// - `const fn`
    /// - `unsafe fn`
    /// - `const unsafe fn`
    /// - `extern fn`
    /// - etc.
    fn parse_fn_front_matter(&mut self)
        -> PResult<'a, (
            Spanned<Constness>,
            Unsafety,
            Spanned<IsAsync>,
            Abi
        )>
    {
        let is_const_fn = self.eat_keyword(kw::Const);
        let const_span = self.prev_span;
        let asyncness = self.parse_asyncness();
        if let IsAsync::Async { .. } = asyncness {
            self.ban_async_in_2015(self.prev_span);
        }
        let asyncness = respan(self.prev_span, asyncness);
        let unsafety = self.parse_unsafety();
        let (constness, unsafety, abi) = if is_const_fn {
            (respan(const_span, Constness::Const), unsafety, Abi::Rust)
        } else {
            let abi = if self.eat_keyword(kw::Extern) {
                self.parse_opt_abi()?.unwrap_or(Abi::C)
            } else {
                Abi::Rust
            };
            (respan(self.prev_span, Constness::NotConst), unsafety, abi)
        };
        if !self.eat_keyword(kw::Fn) {
            // It is possible for `expect_one_of` to recover given the contents of
            // `self.expected_tokens`, therefore, do not use `self.unexpected()` which doesn't
            // account for this.
            if !self.expect_one_of(&[], &[])? { unreachable!() }
        }
        Ok((constness, unsafety, asyncness, abi))
    }

    /// Parses `trait Foo { ... }` or `trait Foo = Bar;`.
    fn parse_item_trait(&mut self, is_auto: IsAuto, unsafety: Unsafety) -> PResult<'a, ItemInfo> {
        let ident = self.parse_ident()?;
        let mut tps = self.parse_generics()?;

        // Parse optional colon and supertrait bounds.
        let bounds = if self.eat(&token::Colon) {
            self.parse_generic_bounds(Some(self.prev_span))?
        } else {
            Vec::new()
        };

        if self.eat(&token::Eq) {
            // it's a trait alias
            let bounds = self.parse_generic_bounds(None)?;
            tps.where_clause = self.parse_where_clause()?;
            self.expect(&token::Semi)?;
            if is_auto == IsAuto::Yes {
                let msg = "trait aliases cannot be `auto`";
                self.struct_span_err(self.prev_span, msg)
                    .span_label(self.prev_span, msg)
                    .emit();
            }
            if unsafety != Unsafety::Normal {
                let msg = "trait aliases cannot be `unsafe`";
                self.struct_span_err(self.prev_span, msg)
                    .span_label(self.prev_span, msg)
                    .emit();
            }
            Ok((ident, ItemKind::TraitAlias(tps, bounds), None))
        } else {
            // it's a normal trait
            tps.where_clause = self.parse_where_clause()?;
            self.expect(&token::OpenDelim(token::Brace))?;
            let mut trait_items = vec![];
            while !self.eat(&token::CloseDelim(token::Brace)) {
                if let token::DocComment(_) = self.token.kind {
                    if self.look_ahead(1,
                    |tok| tok == &token::CloseDelim(token::Brace)) {
                        self.diagnostic().struct_span_err_with_code(
                            self.token.span,
                            "found a documentation comment that doesn't document anything",
                            DiagnosticId::Error("E0584".into()),
                        )
                        .help(
                            "doc comments must come before what they document, maybe a \
                            comment was intended with `//`?",
                        )
                        .emit();
                        self.bump();
                        continue;
                    }
                }
                let mut at_end = false;
                match self.parse_trait_item(&mut at_end) {
                    Ok(item) => trait_items.push(item),
                    Err(mut e) => {
                        e.emit();
                        if !at_end {
                            self.recover_stmt_(SemiColonMode::Break, BlockMode::Break);
                        }
                    }
                }
            }
            Ok((ident, ItemKind::Trait(is_auto, unsafety, tps, bounds, trait_items), None))
        }
    }

    /// Parses the items in a trait declaration.
    pub fn parse_trait_item(&mut self, at_end: &mut bool) -> PResult<'a, TraitItem> {
        maybe_whole!(self, NtTraitItem, |x| x);
        let attrs = self.parse_outer_attributes()?;
        let mut unclosed_delims = vec![];
        let (mut item, tokens) = self.collect_tokens(|this| {
            let item = this.parse_trait_item_(at_end, attrs);
            unclosed_delims.append(&mut this.unclosed_delims);
            item
        })?;
        self.unclosed_delims.append(&mut unclosed_delims);
        // See `parse_item` for why this clause is here.
        if !item.attrs.iter().any(|attr| attr.style == AttrStyle::Inner) {
            item.tokens = Some(tokens);
        }
        Ok(item)
    }

    fn parse_trait_item_(&mut self,
                         at_end: &mut bool,
                         mut attrs: Vec<Attribute>) -> PResult<'a, TraitItem> {
        let lo = self.token.span;
        self.eat_bad_pub();
        let (name, node, generics) = if self.eat_keyword(kw::Type) {
            self.parse_trait_item_assoc_ty()?
        } else if self.is_const_item() {
            self.expect_keyword(kw::Const)?;
            let ident = self.parse_ident()?;
            self.expect(&token::Colon)?;
            let ty = self.parse_ty()?;
            let default = if self.eat(&token::Eq) {
                let expr = self.parse_expr()?;
                self.expect(&token::Semi)?;
                Some(expr)
            } else {
                self.expect(&token::Semi)?;
                None
            };
            (ident, TraitItemKind::Const(ty, default), ast::Generics::default())
        } else if let Some(mac) = self.parse_assoc_macro_invoc("trait", None, &mut false)? {
            // trait item macro.
            (Ident::invalid(), ast::TraitItemKind::Macro(mac), ast::Generics::default())
        } else {
            let (constness, unsafety, asyncness, abi) = self.parse_fn_front_matter()?;

            let ident = self.parse_ident()?;
            let mut generics = self.parse_generics()?;

            let decl = self.parse_fn_decl_with_self(|p: &mut Parser<'a>| {
                // This is somewhat dubious; We don't want to allow
                // argument names to be left off if there is a
                // definition...

                // We don't allow argument names to be left off in edition 2018.
                let is_name_required = p.token.span.rust_2018();
                p.parse_param_general(true, false, |_| is_name_required)
            })?;
            generics.where_clause = self.parse_where_clause()?;

            let sig = ast::MethodSig {
                header: FnHeader {
                    unsafety,
                    constness,
                    abi,
                    asyncness,
                },
                decl,
            };

            let body = match self.token.kind {
                token::Semi => {
                    self.bump();
                    *at_end = true;
                    debug!("parse_trait_methods(): parsing required method");
                    None
                }
                token::OpenDelim(token::Brace) => {
                    debug!("parse_trait_methods(): parsing provided method");
                    *at_end = true;
                    let (inner_attrs, body) = self.parse_inner_attrs_and_block()?;
                    attrs.extend(inner_attrs.iter().cloned());
                    Some(body)
                }
                token::Interpolated(ref nt) => {
                    match **nt {
                        token::NtBlock(..) => {
                            *at_end = true;
                            let (inner_attrs, body) = self.parse_inner_attrs_and_block()?;
                            attrs.extend(inner_attrs.iter().cloned());
                            Some(body)
                        }
                        _ => {
                            return self.expected_semi_or_open_brace();
                        }
                    }
                }
                _ => {
                    return self.expected_semi_or_open_brace();
                }
            };
            (ident, ast::TraitItemKind::Method(sig, body), generics)
        };

        Ok(TraitItem {
            id: ast::DUMMY_NODE_ID,
            ident: name,
            attrs,
            generics,
            node,
            span: lo.to(self.prev_span),
            tokens: None,
        })
    }

    /// Parses the following grammar:
    ///
    ///     TraitItemAssocTy = Ident ["<"...">"] [":" [GenericBounds]] ["where" ...] ["=" Ty]
    fn parse_trait_item_assoc_ty(&mut self)
        -> PResult<'a, (Ident, TraitItemKind, ast::Generics)> {
        let ident = self.parse_ident()?;
        let mut generics = self.parse_generics()?;

        // Parse optional colon and param bounds.
        let bounds = if self.eat(&token::Colon) {
            self.parse_generic_bounds(None)?
        } else {
            Vec::new()
        };
        generics.where_clause = self.parse_where_clause()?;

        let default = if self.eat(&token::Eq) {
            Some(self.parse_ty()?)
        } else {
            None
        };
        self.expect(&token::Semi)?;

        Ok((ident, TraitItemKind::Type(bounds, default), generics))
    }

    /// Parses a `UseTree`.
    ///
    /// ```
    /// USE_TREE = [`::`] `*` |
    ///            [`::`] `{` USE_TREE_LIST `}` |
    ///            PATH `::` `*` |
    ///            PATH `::` `{` USE_TREE_LIST `}` |
    ///            PATH [`as` IDENT]
    /// ```
    fn parse_use_tree(&mut self) -> PResult<'a, UseTree> {
        let lo = self.token.span;

        let mut prefix = ast::Path { segments: Vec::new(), span: lo.shrink_to_lo() };
        let kind = if self.check(&token::OpenDelim(token::Brace)) ||
                      self.check(&token::BinOp(token::Star)) ||
                      self.is_import_coupler() {
            // `use *;` or `use ::*;` or `use {...};` or `use ::{...};`
            let mod_sep_ctxt = self.token.span.ctxt();
            if self.eat(&token::ModSep) {
                prefix.segments.push(
                    PathSegment::path_root(lo.shrink_to_lo().with_ctxt(mod_sep_ctxt))
                );
            }

            if self.eat(&token::BinOp(token::Star)) {
                UseTreeKind::Glob
            } else {
                UseTreeKind::Nested(self.parse_use_tree_list()?)
            }
        } else {
            // `use path::*;` or `use path::{...};` or `use path;` or `use path as bar;`
            prefix = self.parse_path(PathStyle::Mod)?;

            if self.eat(&token::ModSep) {
                if self.eat(&token::BinOp(token::Star)) {
                    UseTreeKind::Glob
                } else {
                    UseTreeKind::Nested(self.parse_use_tree_list()?)
                }
            } else {
                UseTreeKind::Simple(self.parse_rename()?, ast::DUMMY_NODE_ID, ast::DUMMY_NODE_ID)
            }
        };

        Ok(UseTree { prefix, kind, span: lo.to(self.prev_span) })
    }

    /// Parses a `UseTreeKind::Nested(list)`.
    ///
    /// ```
    /// USE_TREE_LIST = Ø | (USE_TREE `,`)* USE_TREE [`,`]
    /// ```
    fn parse_use_tree_list(&mut self) -> PResult<'a, Vec<(UseTree, ast::NodeId)>> {
        self.parse_delim_comma_seq(token::Brace, |p| Ok((p.parse_use_tree()?, ast::DUMMY_NODE_ID)))
            .map(|(r, _)| r)
    }

    fn parse_rename(&mut self) -> PResult<'a, Option<Ident>> {
        if self.eat_keyword(kw::As) {
            self.parse_ident_or_underscore().map(Some)
        } else {
            Ok(None)
        }
    }

    fn parse_ident_or_underscore(&mut self) -> PResult<'a, ast::Ident> {
        match self.token.kind {
            token::Ident(name, false) if name == kw::Underscore => {
                let span = self.token.span;
                self.bump();
                Ok(Ident::new(name, span))
            }
            _ => self.parse_ident(),
        }
    }

    /// Parses `extern crate` links.
    ///
    /// # Examples
    ///
    /// ```
    /// extern crate foo;
    /// extern crate bar as foo;
    /// ```
    fn parse_item_extern_crate(
        &mut self,
        lo: Span,
        visibility: Visibility,
        attrs: Vec<Attribute>
    ) -> PResult<'a, P<Item>> {
        // Accept `extern crate name-like-this` for better diagnostics
        let orig_name = self.parse_crate_name_with_dashes()?;
        let (item_name, orig_name) = if let Some(rename) = self.parse_rename()? {
            (rename, Some(orig_name.name))
        } else {
            (orig_name, None)
        };
        self.expect(&token::Semi)?;

        let span = lo.to(self.prev_span);
        Ok(self.mk_item(span, item_name, ItemKind::ExternCrate(orig_name), visibility, attrs))
    }

    fn parse_crate_name_with_dashes(&mut self) -> PResult<'a, ast::Ident> {
        let error_msg = "crate name using dashes are not valid in `extern crate` statements";
        let suggestion_msg = "if the original crate name uses dashes you need to use underscores \
                              in the code";
        let mut ident = if self.token.is_keyword(kw::SelfLower) {
            self.parse_path_segment_ident()
        } else {
            self.parse_ident()
        }?;
        let mut idents = vec![];
        let mut replacement = vec![];
        let mut fixed_crate_name = false;
        // Accept `extern crate name-like-this` for better diagnostics
        let dash = token::BinOp(token::BinOpToken::Minus);
        if self.token == dash {  // Do not include `-` as part of the expected tokens list
            while self.eat(&dash) {
                fixed_crate_name = true;
                replacement.push((self.prev_span, "_".to_string()));
                idents.push(self.parse_ident()?);
            }
        }
        if fixed_crate_name {
            let fixed_name_sp = ident.span.to(idents.last().unwrap().span);
            let mut fixed_name = format!("{}", ident.name);
            for part in idents {
                fixed_name.push_str(&format!("_{}", part.name));
            }
            ident = Ident::from_str(&fixed_name).with_span_pos(fixed_name_sp);

            self.struct_span_err(fixed_name_sp, error_msg)
                .span_label(fixed_name_sp, "dash-separated idents are not valid")
                .multipart_suggestion(suggestion_msg, replacement, Applicability::MachineApplicable)
                .emit();
        }
        Ok(ident)
    }

    /// Parses an item-position function declaration.
    fn parse_item_fn(
        &mut self,
        unsafety: Unsafety,
        asyncness: Spanned<IsAsync>,
        constness: Spanned<Constness>,
        abi: Abi
    ) -> PResult<'a, ItemInfo> {
        let (ident, mut generics) = self.parse_fn_header()?;
        let allow_c_variadic = abi == Abi::C && unsafety == Unsafety::Unsafe;
        let decl = self.parse_fn_decl(allow_c_variadic)?;
        generics.where_clause = self.parse_where_clause()?;
        let (inner_attrs, body) = self.parse_inner_attrs_and_block()?;
        let header = FnHeader { unsafety, asyncness, constness, abi };
        Ok((ident, ItemKind::Fn(decl, header, generics, body), Some(inner_attrs)))
    }

    /// Parses the name and optional generic types of a function header.
    fn parse_fn_header(&mut self) -> PResult<'a, (Ident, ast::Generics)> {
        let id = self.parse_ident()?;
        let generics = self.parse_generics()?;
        Ok((id, generics))
    }

    /// Parses the argument list and result type of a function declaration.
    fn parse_fn_decl(&mut self, allow_c_variadic: bool) -> PResult<'a, P<FnDecl>> {
        let (args, c_variadic) = self.parse_fn_params(true, allow_c_variadic)?;
        let ret_ty = self.parse_ret_ty(true)?;

        Ok(P(FnDecl {
            inputs: args,
            output: ret_ty,
            c_variadic,
        }))
    }

    /// Parses `extern` for foreign ABIs modules.
    ///
    /// `extern` is expected to have been
    /// consumed before calling this method.
    ///
    /// # Examples
    ///
    /// ```ignore (only-for-syntax-highlight)
    /// extern "C" {}
    /// extern {}
    /// ```
    fn parse_item_foreign_mod(
        &mut self,
        lo: Span,
        opt_abi: Option<Abi>,
        visibility: Visibility,
        mut attrs: Vec<Attribute>,
        extern_sp: Span,
    ) -> PResult<'a, P<Item>> {
        self.expect(&token::OpenDelim(token::Brace))?;

        let abi = opt_abi.unwrap_or(Abi::C);

        attrs.extend(self.parse_inner_attributes()?);

        let mut foreign_items = vec![];
        while !self.eat(&token::CloseDelim(token::Brace)) {
            foreign_items.push(self.parse_foreign_item(extern_sp)?);
        }

        let prev_span = self.prev_span;
        let m = ast::ForeignMod {
            abi,
            items: foreign_items
        };
        let invalid = Ident::invalid();
        Ok(self.mk_item(lo.to(prev_span), invalid, ItemKind::ForeignMod(m), visibility, attrs))
    }

    /// Parses a foreign item.
    crate fn parse_foreign_item(&mut self, extern_sp: Span) -> PResult<'a, ForeignItem> {
        maybe_whole!(self, NtForeignItem, |ni| ni);

        let attrs = self.parse_outer_attributes()?;
        let lo = self.token.span;
        let visibility = self.parse_visibility(false)?;

        // FOREIGN STATIC ITEM
        // Treat `const` as `static` for error recovery, but don't add it to expected tokens.
        if self.check_keyword(kw::Static) || self.token.is_keyword(kw::Const) {
            if self.token.is_keyword(kw::Const) {
                self.diagnostic()
                    .struct_span_err(self.token.span, "extern items cannot be `const`")
                    .span_suggestion(
                        self.token.span,
                        "try using a static value",
                        "static".to_owned(),
                        Applicability::MachineApplicable
                    ).emit();
            }
            self.bump(); // `static` or `const`
            return Ok(self.parse_item_foreign_static(visibility, lo, attrs)?);
        }
        // FOREIGN FUNCTION ITEM
        if self.check_keyword(kw::Fn) {
            return Ok(self.parse_item_foreign_fn(visibility, lo, attrs, extern_sp)?);
        }
        // FOREIGN TYPE ITEM
        if self.check_keyword(kw::Type) {
            return Ok(self.parse_item_foreign_type(visibility, lo, attrs)?);
        }

        match self.parse_assoc_macro_invoc("extern", Some(&visibility), &mut false)? {
            Some(mac) => {
                Ok(
                    ForeignItem {
                        ident: Ident::invalid(),
                        span: lo.to(self.prev_span),
                        id: ast::DUMMY_NODE_ID,
                        attrs,
                        vis: visibility,
                        node: ForeignItemKind::Macro(mac),
                    }
                )
            }
            None => {
                if !attrs.is_empty()  {
                    self.expected_item_err(&attrs)?;
                }

                self.unexpected()
            }
        }
    }

    /// Parses a function declaration from a foreign module.
    fn parse_item_foreign_fn(
        &mut self,
        vis: ast::Visibility,
        lo: Span,
        attrs: Vec<Attribute>,
        extern_sp: Span,
    ) -> PResult<'a, ForeignItem> {
        self.expect_keyword(kw::Fn)?;

        let (ident, mut generics) = self.parse_fn_header()?;
        let decl = self.parse_fn_decl(true)?;
        generics.where_clause = self.parse_where_clause()?;
        let hi = self.token.span;
        self.parse_semi_or_incorrect_foreign_fn_body(&ident, extern_sp)?;
        Ok(ast::ForeignItem {
            ident,
            attrs,
            node: ForeignItemKind::Fn(decl, generics),
            id: ast::DUMMY_NODE_ID,
            span: lo.to(hi),
            vis,
        })
    }

    /// Parses a static item from a foreign module.
    /// Assumes that the `static` keyword is already parsed.
    fn parse_item_foreign_static(&mut self, vis: ast::Visibility, lo: Span, attrs: Vec<Attribute>)
                                 -> PResult<'a, ForeignItem> {
        let mutbl = self.parse_mutability();
        let ident = self.parse_ident()?;
        self.expect(&token::Colon)?;
        let ty = self.parse_ty()?;
        let hi = self.token.span;
        self.expect(&token::Semi)?;
        Ok(ForeignItem {
            ident,
            attrs,
            node: ForeignItemKind::Static(ty, mutbl),
            id: ast::DUMMY_NODE_ID,
            span: lo.to(hi),
            vis,
        })
    }

    /// Parses a type from a foreign module.
    fn parse_item_foreign_type(&mut self, vis: ast::Visibility, lo: Span, attrs: Vec<Attribute>)
                             -> PResult<'a, ForeignItem> {
        self.expect_keyword(kw::Type)?;

        let ident = self.parse_ident()?;
        let hi = self.token.span;
        self.expect(&token::Semi)?;
        Ok(ast::ForeignItem {
            ident,
            attrs,
            node: ForeignItemKind::Ty,
            id: ast::DUMMY_NODE_ID,
            span: lo.to(hi),
            vis
        })
    }

    fn is_static_global(&mut self) -> bool {
        if self.check_keyword(kw::Static) {
            // Check if this could be a closure
            !self.look_ahead(1, |token| {
                if token.is_keyword(kw::Move) {
                    return true;
                }
                match token.kind {
                    token::BinOp(token::Or) | token::OrOr => true,
                    _ => false,
                }
            })
        } else {
            false
        }
    }

    fn parse_item_const(&mut self, m: Option<Mutability>) -> PResult<'a, ItemInfo> {
        let id = if m.is_none() { self.parse_ident_or_underscore() } else { self.parse_ident() }?;
        self.expect(&token::Colon)?;
        let ty = self.parse_ty()?;
        self.expect(&token::Eq)?;
        let e = self.parse_expr()?;
        self.expect(&token::Semi)?;
        let item = match m {
            Some(m) => ItemKind::Static(ty, m, e),
            None => ItemKind::Const(ty, e),
        };
        Ok((id, item, None))
    }

    /// Parses `type Foo = Bar;` or returns `None`
    /// without modifying the parser state.
    fn eat_type(&mut self) -> Option<PResult<'a, (Ident, AliasKind, ast::Generics)>> {
        // This parses the grammar:
        //     Ident ["<"...">"] ["where" ...] ("=" | ":") Ty ";"
        if self.eat_keyword(kw::Type) {
            Some(self.parse_type_alias())
        } else {
            None
        }
    }

    /// Parses a type alias or opaque type.
    fn parse_type_alias(&mut self) -> PResult<'a, (Ident, AliasKind, ast::Generics)> {
        let ident = self.parse_ident()?;
        let mut tps = self.parse_generics()?;
        tps.where_clause = self.parse_where_clause()?;
        self.expect(&token::Eq)?;
        let alias = if self.check_keyword(kw::Impl) {
            self.bump();
            let bounds = self.parse_generic_bounds(Some(self.prev_span))?;
            AliasKind::OpaqueTy(bounds)
        } else {
            let ty = self.parse_ty()?;
            AliasKind::Weak(ty)
        };
        self.expect(&token::Semi)?;
        Ok((ident, alias, tps))
    }

    /// Parses an enum declaration.
    fn parse_item_enum(&mut self) -> PResult<'a, ItemInfo> {
        let id = self.parse_ident()?;
        let mut generics = self.parse_generics()?;
        generics.where_clause = self.parse_where_clause()?;
        self.expect(&token::OpenDelim(token::Brace))?;

        let enum_definition = self.parse_enum_def(&generics).map_err(|e| {
            self.recover_stmt();
            self.eat(&token::CloseDelim(token::Brace));
            e
        })?;
        Ok((id, ItemKind::Enum(enum_definition, generics), None))
    }

    /// Parses the part of an enum declaration following the `{`.
    fn parse_enum_def(&mut self, _generics: &ast::Generics) -> PResult<'a, EnumDef> {
        let mut variants = Vec::new();
        while self.token != token::CloseDelim(token::Brace) {
            let variant_attrs = self.parse_outer_attributes()?;
            let vlo = self.token.span;

            self.eat_bad_pub();
            let ident = self.parse_ident()?;

            let struct_def = if self.check(&token::OpenDelim(token::Brace)) {
                // Parse a struct variant.
                let (fields, recovered) = self.parse_record_struct_body()?;
                VariantData::Struct(fields, recovered)
            } else if self.check(&token::OpenDelim(token::Paren)) {
                VariantData::Tuple(
                    self.parse_tuple_struct_body()?,
                    ast::DUMMY_NODE_ID,
                )
            } else {
                VariantData::Unit(ast::DUMMY_NODE_ID)
            };

            let disr_expr = if self.eat(&token::Eq) {
                Some(AnonConst {
                    id: ast::DUMMY_NODE_ID,
                    value: self.parse_expr()?,
                })
            } else {
                None
            };

            let vr = ast::Variant {
                ident,
                id: ast::DUMMY_NODE_ID,
                attrs: variant_attrs,
                data: struct_def,
                disr_expr,
                span: vlo.to(self.prev_span),
            };
            variants.push(vr);

            if !self.eat(&token::Comma) {
                if self.token.is_ident() && !self.token.is_reserved_ident() {
                    let sp = self.sess.source_map().next_point(self.prev_span);
                    self.struct_span_err(sp, "missing comma")
                        .span_suggestion_short(
                            sp,
                            "missing comma",
                            ",".to_owned(),
                            Applicability::MaybeIncorrect,
                        )
                        .emit();
                } else {
                    break;
                }
            }
        }
        self.expect(&token::CloseDelim(token::Brace))?;

        Ok(ast::EnumDef { variants })
    }

    /// Parses `struct Foo { ... }`.
    fn parse_item_struct(&mut self) -> PResult<'a, ItemInfo> {
        let class_name = self.parse_ident()?;

        let mut generics = self.parse_generics()?;

        // There is a special case worth noting here, as reported in issue #17904.
        // If we are parsing a tuple struct it is the case that the where clause
        // should follow the field list. Like so:
        //
        // struct Foo<T>(T) where T: Copy;
        //
        // If we are parsing a normal record-style struct it is the case
        // that the where clause comes before the body, and after the generics.
        // So if we look ahead and see a brace or a where-clause we begin
        // parsing a record style struct.
        //
        // Otherwise if we look ahead and see a paren we parse a tuple-style
        // struct.

        let vdata = if self.token.is_keyword(kw::Where) {
            generics.where_clause = self.parse_where_clause()?;
            if self.eat(&token::Semi) {
                // If we see a: `struct Foo<T> where T: Copy;` style decl.
                VariantData::Unit(ast::DUMMY_NODE_ID)
            } else {
                // If we see: `struct Foo<T> where T: Copy { ... }`
                let (fields, recovered) = self.parse_record_struct_body()?;
                VariantData::Struct(fields, recovered)
            }
        // No `where` so: `struct Foo<T>;`
        } else if self.eat(&token::Semi) {
            VariantData::Unit(ast::DUMMY_NODE_ID)
        // Record-style struct definition
        } else if self.token == token::OpenDelim(token::Brace) {
            let (fields, recovered) = self.parse_record_struct_body()?;
            VariantData::Struct(fields, recovered)
        // Tuple-style struct definition with optional where-clause.
        } else if self.token == token::OpenDelim(token::Paren) {
            let body = VariantData::Tuple(self.parse_tuple_struct_body()?, ast::DUMMY_NODE_ID);
            generics.where_clause = self.parse_where_clause()?;
            self.expect(&token::Semi)?;
            body
        } else {
            let token_str = self.this_token_descr();
            let mut err = self.fatal(&format!(
                "expected `where`, `{{`, `(`, or `;` after struct name, found {}",
                token_str
            ));
            err.span_label(self.token.span, "expected `where`, `{`, `(`, or `;` after struct name");
            return Err(err);
        };

        Ok((class_name, ItemKind::Struct(vdata, generics), None))
    }

    /// Parses `union Foo { ... }`.
    fn parse_item_union(&mut self) -> PResult<'a, ItemInfo> {
        let class_name = self.parse_ident()?;

        let mut generics = self.parse_generics()?;

        let vdata = if self.token.is_keyword(kw::Where) {
            generics.where_clause = self.parse_where_clause()?;
            let (fields, recovered) = self.parse_record_struct_body()?;
            VariantData::Struct(fields, recovered)
        } else if self.token == token::OpenDelim(token::Brace) {
            let (fields, recovered) = self.parse_record_struct_body()?;
            VariantData::Struct(fields, recovered)
        } else {
            let token_str = self.this_token_descr();
            let mut err = self.fatal(&format!(
                "expected `where` or `{{` after union name, found {}", token_str));
            err.span_label(self.token.span, "expected `where` or `{` after union name");
            return Err(err);
        };

        Ok((class_name, ItemKind::Union(vdata, generics), None))
    }

    pub(super) fn is_union_item(&self) -> bool {
        self.token.is_keyword(kw::Union) &&
        self.look_ahead(1, |t| t.is_ident() && !t.is_reserved_ident())
    }

    fn parse_record_struct_body(
        &mut self,
    ) -> PResult<'a, (Vec<StructField>, /* recovered */ bool)> {
        let mut fields = Vec::new();
        let mut recovered = false;
        if self.eat(&token::OpenDelim(token::Brace)) {
            while self.token != token::CloseDelim(token::Brace) {
                let field = self.parse_struct_decl_field().map_err(|e| {
                    self.recover_stmt();
                    recovered = true;
                    e
                });
                match field {
                    Ok(field) => fields.push(field),
                    Err(mut err) => {
                        err.emit();
                    }
                }
            }
            self.eat(&token::CloseDelim(token::Brace));
        } else {
            let token_str = self.this_token_descr();
            let mut err = self.fatal(&format!(
                    "expected `where`, or `{{` after struct name, found {}", token_str));
            err.span_label(self.token.span, "expected `where`, or `{` after struct name");
            return Err(err);
        }

        Ok((fields, recovered))
    }

    fn parse_tuple_struct_body(&mut self) -> PResult<'a, Vec<StructField>> {
        // This is the case where we find `struct Foo<T>(T) where T: Copy;`
        // Unit like structs are handled in parse_item_struct function
        self.parse_paren_comma_seq(|p| {
            let attrs = p.parse_outer_attributes()?;
            let lo = p.token.span;
            let vis = p.parse_visibility(true)?;
            let ty = p.parse_ty()?;
            Ok(StructField {
                span: lo.to(ty.span),
                vis,
                ident: None,
                id: ast::DUMMY_NODE_ID,
                ty,
                attrs,
            })
        }).map(|(r, _)| r)
    }

    /// Parses an element of a struct declaration.
    fn parse_struct_decl_field(&mut self) -> PResult<'a, StructField> {
        let attrs = self.parse_outer_attributes()?;
        let lo = self.token.span;
        let vis = self.parse_visibility(false)?;
        self.parse_single_struct_field(lo, vis, attrs)
    }

    /// Parses a structure field declaration.
    fn parse_single_struct_field(&mut self,
                                     lo: Span,
                                     vis: Visibility,
                                     attrs: Vec<Attribute> )
                                     -> PResult<'a, StructField> {
        let mut seen_comma: bool = false;
        let a_var = self.parse_name_and_ty(lo, vis, attrs)?;
        if self.token == token::Comma {
            seen_comma = true;
        }
        match self.token.kind {
            token::Comma => {
                self.bump();
            }
            token::CloseDelim(token::Brace) => {}
            token::DocComment(_) => {
                let previous_span = self.prev_span;
                let mut err = self.span_fatal_err(self.token.span, Error::UselessDocComment);
                self.bump(); // consume the doc comment
                let comma_after_doc_seen = self.eat(&token::Comma);
                // `seen_comma` is always false, because we are inside doc block
                // condition is here to make code more readable
                if seen_comma == false && comma_after_doc_seen == true {
                    seen_comma = true;
                }
                if comma_after_doc_seen || self.token == token::CloseDelim(token::Brace) {
                    err.emit();
                } else {
                    if seen_comma == false {
                        let sp = self.sess.source_map().next_point(previous_span);
                        err.span_suggestion(
                            sp,
                            "missing comma here",
                            ",".into(),
                            Applicability::MachineApplicable
                        );
                    }
                    return Err(err);
                }
            }
            _ => {
                let sp = self.sess.source_map().next_point(self.prev_span);
                let mut err = self.struct_span_err(sp, &format!("expected `,`, or `}}`, found {}",
                                                                self.this_token_descr()));
                if self.token.is_ident() {
                    // This is likely another field; emit the diagnostic and keep going
                    err.span_suggestion(
                        sp,
                        "try adding a comma",
                        ",".into(),
                        Applicability::MachineApplicable,
                    );
                    err.emit();
                } else {
                    return Err(err)
                }
            }
        }
        Ok(a_var)
    }

    /// Parses a structure field.
    fn parse_name_and_ty(
        &mut self,
        lo: Span,
        vis: Visibility,
        attrs: Vec<Attribute>
    ) -> PResult<'a, StructField> {
        let name = self.parse_ident()?;
        self.expect(&token::Colon)?;
        let ty = self.parse_ty()?;
        Ok(StructField {
            span: lo.to(self.prev_span),
            ident: Some(name),
            vis,
            id: ast::DUMMY_NODE_ID,
            ty,
            attrs,
        })
    }

    pub(super) fn eat_macro_def(
        &mut self,
        attrs: &[Attribute],
        vis: &Visibility,
        lo: Span
    ) -> PResult<'a, Option<P<Item>>> {
        let token_lo = self.token.span;
        let (ident, def) = if self.eat_keyword(kw::Macro) {
            let ident = self.parse_ident()?;
            let tokens = if self.check(&token::OpenDelim(token::Brace)) {
                match self.parse_token_tree() {
                    TokenTree::Delimited(_, _, tts) => tts,
                    _ => unreachable!(),
                }
            } else if self.check(&token::OpenDelim(token::Paren)) {
                let args = self.parse_token_tree();
                let body = if self.check(&token::OpenDelim(token::Brace)) {
                    self.parse_token_tree()
                } else {
                    self.unexpected()?;
                    unreachable!()
                };
                TokenStream::new(vec![
                    args.into(),
                    TokenTree::token(token::FatArrow, token_lo.to(self.prev_span)).into(),
                    body.into(),
                ])
            } else {
                self.unexpected()?;
                unreachable!()
            };

            (ident, ast::MacroDef { tokens: tokens.into(), legacy: false })
        } else if self.check_keyword(sym::macro_rules) &&
                  self.look_ahead(1, |t| *t == token::Not) &&
                  self.look_ahead(2, |t| t.is_ident()) {
            let prev_span = self.prev_span;
            self.complain_if_pub_macro(&vis.node, prev_span);
            self.bump();
            self.bump();

            let ident = self.parse_ident()?;
            let (delim, tokens) = self.expect_delimited_token_tree()?;
            if delim != MacDelimiter::Brace && !self.eat(&token::Semi) {
                self.report_invalid_macro_expansion_item();
            }

            (ident, ast::MacroDef { tokens, legacy: true })
        } else {
            return Ok(None);
        };

        let span = lo.to(self.prev_span);
        Ok(Some(self.mk_item(span, ident, ItemKind::MacroDef(def), vis.clone(), attrs.to_vec())))
    }

    fn complain_if_pub_macro(&self, vis: &VisibilityKind, sp: Span) {
        match *vis {
            VisibilityKind::Inherited => {}
            _ => {
                let mut err = if self.token.is_keyword(sym::macro_rules) {
                    let mut err = self.diagnostic()
                        .struct_span_err(sp, "can't qualify macro_rules invocation with `pub`");
                    err.span_suggestion(
                        sp,
                        "try exporting the macro",
                        "#[macro_export]".to_owned(),
                        Applicability::MaybeIncorrect // speculative
                    );
                    err
                } else {
                    let mut err = self.diagnostic()
                        .struct_span_err(sp, "can't qualify macro invocation with `pub`");
                    err.help("try adjusting the macro to put `pub` inside the invocation");
                    err
                };
                err.emit();
            }
        }
    }

    fn mk_item(&self, span: Span, ident: Ident, node: ItemKind, vis: Visibility,
               attrs: Vec<Attribute>) -> P<Item> {
        P(Item {
            ident,
            attrs,
            id: ast::DUMMY_NODE_ID,
            node,
            vis,
            span,
            tokens: None,
        })
    }
}
