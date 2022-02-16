use crate::deriving::generic::ty::*;
use crate::deriving::generic::*;

use rustc_ast::ptr::P;
use rustc_ast::walk_list;
use rustc_ast::EnumDef;
use rustc_ast::VariantData;
use rustc_ast::{Expr, MetaItem};
use rustc_errors::Applicability;
use rustc_expand::base::{Annotatable, DummyResult, ExtCtxt};
use rustc_span::symbol::Ident;
use rustc_span::symbol::{kw, sym};
use rustc_span::Span;
use smallvec::SmallVec;

pub fn expand_deriving_default(
    cx: &mut ExtCtxt<'_>,
    span: Span,
    mitem: &MetaItem,
    item: &Annotatable,
    push: &mut dyn FnMut(Annotatable),
) {
    item.visit_with(&mut DetectNonVariantDefaultAttr { cx });

    let inline = cx.meta_word(span, sym::inline);
    let attrs = vec![cx.attribute(inline)];
    let trait_def = TraitDef {
        span,
        attributes: Vec::new(),
        path: Path::new(vec![kw::Default, sym::Default]),
        additional_bounds: Vec::new(),
        generics: Bounds::empty(),
        is_unsafe: false,
        supports_unions: false,
        methods: vec![MethodDef {
            name: kw::Default,
            generics: Bounds::empty(),
            explicit_self: None,
            args: Vec::new(),
            ret_ty: Self_,
            attributes: attrs,
            is_unsafe: false,
            unify_fieldless_variants: false,
            combine_substructure: combine_substructure(Box::new(|cx, trait_span, substr| {
                match substr.fields {
                    StaticStruct(_, fields) => {
                        default_struct_substructure(cx, trait_span, substr, fields)
                    }
                    StaticEnum(enum_def, _) => {
                        if !cx.sess.features_untracked().derive_default_enum {
                            rustc_session::parse::feature_err(
                                cx.parse_sess(),
                                sym::derive_default_enum,
                                span,
                                "deriving `Default` on enums is experimental",
                            )
                            .emit();
                        }
                        default_enum_substructure(cx, trait_span, enum_def)
                    }
                    _ => cx.span_bug(trait_span, "method in `derive(Default)`"),
                }
            })),
        }],
        associated_types: Vec::new(),
    };
    trait_def.expand(cx, mitem, item, push)
}

fn default_struct_substructure(
    cx: &mut ExtCtxt<'_>,
    trait_span: Span,
    substr: &Substructure<'_>,
    summary: &StaticFields,
) -> P<Expr> {
    // Note that `kw::Default` is "default" and `sym::Default` is "Default"!
    let default_ident = cx.std_path(&[kw::Default, sym::Default, kw::Default]);
    let default_call = |span| cx.expr_call_global(span, default_ident.clone(), Vec::new());

    match summary {
        Unnamed(ref fields, is_tuple) => {
            if !is_tuple {
                cx.expr_ident(trait_span, substr.type_ident)
            } else {
                let exprs = fields.iter().map(|sp| default_call(*sp)).collect();
                cx.expr_call_ident(trait_span, substr.type_ident, exprs)
            }
        }
        Named(ref fields) => {
            let default_fields = fields
                .iter()
                .map(|&(ident, span)| cx.field_imm(span, ident, default_call(span)))
                .collect();
            cx.expr_struct_ident(trait_span, substr.type_ident, default_fields)
        }
    }
}

fn default_enum_substructure(
    cx: &mut ExtCtxt<'_>,
    trait_span: Span,
    enum_def: &EnumDef,
) -> P<Expr> {
    let default_variant = match extract_default_variant(cx, enum_def, trait_span) {
        Ok(value) => value,
        Err(()) => return DummyResult::raw_expr(trait_span, true),
    };

    // At this point, we know that there is exactly one variant with a `#[default]` attribute. The
    // attribute hasn't yet been validated.

    if let Err(()) = validate_default_attribute(cx, default_variant) {
        return DummyResult::raw_expr(trait_span, true);
    }

    // We now know there is exactly one unit variant with exactly one `#[default]` attribute.

    cx.expr_path(cx.path(
        default_variant.span,
        vec![Ident::new(kw::SelfUpper, default_variant.span), default_variant.ident],
    ))
}

fn extract_default_variant<'a>(
    cx: &mut ExtCtxt<'_>,
    enum_def: &'a EnumDef,
    trait_span: Span,
) -> Result<&'a rustc_ast::Variant, ()> {
    let default_variants: SmallVec<[_; 1]> = enum_def
        .variants
        .iter()
        .filter(|variant| cx.sess.contains_name(&variant.attrs, kw::Default))
        .collect();

    let variant = match default_variants.as_slice() {
        [variant] => variant,
        [] => {
            let possible_defaults = enum_def
                .variants
                .iter()
                .filter(|variant| matches!(variant.data, VariantData::Unit(..)))
                .filter(|variant| !cx.sess.contains_name(&variant.attrs, sym::non_exhaustive));

            let mut diag = cx.struct_span_err(trait_span, "no default declared");
            diag.help("make a unit variant default by placing `#[default]` above it");
            for variant in possible_defaults {
                // Suggest making each unit variant default.
                diag.tool_only_span_suggestion(
                    variant.span,
                    &format!("make `{}` default", variant.ident),
                    format!("#[default] {}", variant.ident),
                    Applicability::MaybeIncorrect,
                );
            }
            diag.emit();

            return Err(());
        }
        [first, rest @ ..] => {
            let mut diag = cx.struct_span_err(trait_span, "multiple declared defaults");
            diag.span_label(first.span, "first default");
            diag.span_labels(rest.iter().map(|variant| variant.span), "additional default");
            diag.note("only one variant can be default");
            for variant in &default_variants {
                // Suggest making each variant already tagged default.
                let suggestion = default_variants
                    .iter()
                    .filter_map(|v| {
                        if v.ident == variant.ident {
                            None
                        } else {
                            Some((cx.sess.find_by_name(&v.attrs, kw::Default)?.span, String::new()))
                        }
                    })
                    .collect();

                diag.tool_only_multipart_suggestion(
                    &format!("make `{}` default", variant.ident),
                    suggestion,
                    Applicability::MaybeIncorrect,
                );
            }
            diag.emit();

            return Err(());
        }
    };

    if !matches!(variant.data, VariantData::Unit(..)) {
        cx.struct_span_err(
            variant.ident.span,
            "the `#[default]` attribute may only be used on unit enum variants",
        )
        .help("consider a manual implementation of `Default`")
        .emit();

        return Err(());
    }

    if let Some(non_exhaustive_attr) = cx.sess.find_by_name(&variant.attrs, sym::non_exhaustive) {
        cx.struct_span_err(variant.ident.span, "default variant must be exhaustive")
            .span_label(non_exhaustive_attr.span, "declared `#[non_exhaustive]` here")
            .help("consider a manual implementation of `Default`")
            .emit();

        return Err(());
    }

    Ok(variant)
}

fn validate_default_attribute(
    cx: &mut ExtCtxt<'_>,
    default_variant: &rustc_ast::Variant,
) -> Result<(), ()> {
    let attrs: SmallVec<[_; 1]> =
        cx.sess.filter_by_name(&default_variant.attrs, kw::Default).collect();

    let attr = match attrs.as_slice() {
        [attr] => attr,
        [] => cx.bug(
            "this method must only be called with a variant that has a `#[default]` attribute",
        ),
        [first, rest @ ..] => {
            let suggestion_text =
                if rest.len() == 1 { "try removing this" } else { "try removing these" };

            cx.struct_span_err(default_variant.ident.span, "multiple `#[default]` attributes")
                .note("only one `#[default]` attribute is needed")
                .span_label(first.span, "`#[default]` used here")
                .span_label(rest[0].span, "`#[default]` used again here")
                .span_help(rest.iter().map(|attr| attr.span).collect::<Vec<_>>(), suggestion_text)
                // This would otherwise display the empty replacement, hence the otherwise
                // repetitive `.span_help` call above.
                .tool_only_multipart_suggestion(
                    suggestion_text,
                    rest.iter().map(|attr| (attr.span, String::new())).collect(),
                    Applicability::MachineApplicable,
                )
                .emit();

            return Err(());
        }
    };
    if !attr.is_word() {
        cx.struct_span_err(attr.span, "`#[default]` attribute does not accept a value")
            .span_suggestion_hidden(
                attr.span,
                "try using `#[default]`",
                "#[default]".into(),
                Applicability::MaybeIncorrect,
            )
            .emit();

        return Err(());
    }
    Ok(())
}

struct DetectNonVariantDefaultAttr<'a, 'b> {
    cx: &'a ExtCtxt<'b>,
}

impl<'a, 'b> rustc_ast::visit::Visitor<'a> for DetectNonVariantDefaultAttr<'a, 'b> {
    fn visit_attribute(&mut self, attr: &'a rustc_ast::Attribute) {
        if attr.has_name(kw::Default) {
            self.cx
                .struct_span_err(
                    attr.span,
                    "the `#[default]` attribute may only be used on unit enum variants",
                )
                .emit();
        }

        rustc_ast::visit::walk_attribute(self, attr);
    }
    fn visit_variant(&mut self, v: &'a rustc_ast::Variant) {
        self.visit_ident(v.ident);
        self.visit_vis(&v.vis);
        self.visit_variant_data(&v.data);
        walk_list!(self, visit_anon_const, &v.disr_expr);
        for attr in &v.attrs {
            rustc_ast::visit::walk_attribute(self, attr);
        }
    }
}
