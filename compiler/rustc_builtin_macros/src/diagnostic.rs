#![allow(warnings)]

use std::convert::identity;

use rustc_ast::{AttrArgs, AttrStyle, Item, ItemKind, Trait, ast};
use rustc_attr_parsing::parser::{AllowExprMetavar, ArgParser};
use rustc_attr_parsing::{AttributeParser, AttributeSafety, ParsedDescription, ShouldEmit};
use rustc_expand::base::{Annotatable, ExtCtxt};
use rustc_feature::template;
use rustc_hir::{AttrPath, Target};
use rustc_parse::parser::Recovery;
use rustc_span::{Span, sym};

pub(crate) mod rustc_on_unimplemented {
    use super::*;
    pub(crate) fn expand(ecx: &mut ExtCtxt<'_>, attr: &ast::Attribute, item: &mut Annotatable) {
        let attr = attr.get_normal_item();
        let span = attr.span();
        let Some(args) = ArgParser::from_attr_args(
            &attr.args.unparsed_ref().unwrap(),
            &[sym::diagnostic, sym::rustc_on_unimplemented],
            &ecx.sess.psess,
            ShouldEmit::ErrorsAndLints { recovery: Recovery::Forbidden },
            AllowExprMetavar::No,
        ) else {
            // Lints/errors are/will be emitted by ArgParser.
            return;
        };

        match item {
            Annotatable::Item(Item {
                span: target_span,
                kind: ItemKind::Trait(Trait { on_unimplemented, .. }),
                ..
            }) => {
                if on_unimplemented.is_some() {
                    // FIXME(mejrs) coalescing multiple rustc_on_unimplemented attrs isn't
                    // (and was never) supported - might be nice to have at some point
                    ecx.dcx().span_err(
                        span,
                        "using multiple `#[diagnostic::rustc_on_unimplemented]` is not supported",
                    );
                }
                *on_unimplemented = AttributeParser::parse_single_args(
                    ecx.sess,
                    span,
                    span,
                    AttrStyle::Inner,
                    AttrPath::from_ast(&attr.path, identity),
                    None,
                    AttributeSafety::Normal,
                    ParsedDescription::Macro,
                    *target_span,
                    ecx.current_expansion.lint_node_id,
                    Target::Trait,
                    Some(ecx.ecfg.features),
                    ShouldEmit::ErrorsAndLints { recovery: Recovery::Forbidden },
                    &args,
                    rustc_attr_parsing::parse_rustc_on_unimplemented,
                    &template!(List: &[r#"/*opt*/ message = "...", /*opt*/ label = "...", /*opt*/ note = "...""#]),
                );
            }
            _ => {
                ecx.dcx()
                    .span_err(item.span(), "`#[diagnostic::rustc_on_unimplemented]` is only supported on trait definitions");
            }
        }
    }
}
