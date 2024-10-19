use rustc_ast as ast;
use rustc_errors::DiagCtxtHandle;
use rustc_hir::attribute::ParsedAttributeKind;
use rustc_span::{ErrorGuaranteed, Symbol, sym};

use crate::MaybeParsedAttribute;

pub struct AttributeParseContext<'dcx> {
    tools: Vec<Symbol>,
    dcx: DiagCtxtHandle<'dcx>,
}

impl<'dcx> AttributeParseContext<'dcx> {
    pub fn new(dcx: DiagCtxtHandle<'dcx>, tools: Vec<Symbol>) -> Self {
        Self { dcx, tools }
    }

    pub fn parse_attribute_list<'a>(
        &'a self,
        attrs: &'a [ast::Attribute],
    ) -> impl Iterator<Item = (MaybeParsedAttribute<'a>, &'a ast::Attribute)> + use<'a> {
        attrs
            .iter()
            .map(move |attr| (self.parse_attribute(attr), attr))
            .filter_map(|(r, a)| Some((r.ok()?, a)))
    }

    /// Parses an attribute, if it can.
    ///
    /// All attributes go through here to be parsed.
    /// This function should return [`MaybeParsedAttribute::MustRemainUnparsed`] as little as possible,
    /// because any time it does it implies that the attribute needs to be parsed somewhere else while
    /// what we want is to centralize parsing.
    ///
    /// Only [custom tool attributes](https://github.com/rust-lang/rust/issues/66079) can definitely not
    /// be parsed and should remain unparsed. For any other attribute you better have a very good reason
    /// not to parse it here.
    pub fn parse_attribute<'a>(
        &self,
        attr: &'a ast::Attribute,
    ) -> Result<MaybeParsedAttribute<'a>, ErrorGuaranteed> {
        let res = match &attr.kind {
            ast::AttrKind::DocComment(comment_kind, symbol) => MaybeParsedAttribute::Parsed(
                ParsedAttributeKind::DocComment(*comment_kind, *symbol),
            ),
            ast::AttrKind::Normal(n) => {
                const FIXME_TEMPORARY_ATTR_ALLOWLIST: &[Symbol] = &[sym::cfg];

                // if we're here, we must be compiling a tool attribute... Or someone forgot to
                // parse their fancy new attribute. Let's warn them in any case. If you are that
                // person, and you really your attribute should remain unparsed, carefully read the
                // documentation in this module and if you still think so you can add an exception
                // to this assertion.
                let attr_sym = n.item.path.segments.first().unwrap().ident.name;
                assert!(
                    self.tools.contains(&attr_sym)
                        || FIXME_TEMPORARY_ATTR_ALLOWLIST.contains(&attr_sym),
                    "attribute {attr_sym} wasn't parsed and isn't a know tool attribute"
                );

                MaybeParsedAttribute::MustRemainUnparsed(&*n)
            }
        };

        Ok(res)
    }
}
