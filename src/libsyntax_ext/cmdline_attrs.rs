//! Attributes injected into the crate root from command line using `-Z crate-attr`.

use syntax::ast::{self, AttrItem, AttrStyle};
use syntax::attr::mk_attr;
use syntax::panictry;
use syntax::parse::{self, token};
use syntax::sess::ParseSess;
use syntax_pos::FileName;

pub fn inject(mut krate: ast::Crate, parse_sess: &ParseSess, attrs: &[String]) -> ast::Crate {
    for raw_attr in attrs {
        let mut parser = parse::new_parser_from_source_str(
            parse_sess,
            FileName::cli_crate_attr_source_code(&raw_attr),
            raw_attr.clone(),
        );

        let start_span = parser.token.span;
        let AttrItem { path, tokens } = panictry!(parser.parse_attr_item());
        let end_span = parser.token.span;
        if parser.token != token::Eof {
            parse_sess.span_diagnostic
                .span_err(start_span.to(end_span), "invalid crate attribute");
            continue;
        }

        krate.attrs.push(mk_attr(AttrStyle::Inner, path, tokens, start_span.to(end_span)));
    }

    krate
}
