//! Attributes injected into the crate root from command line using `-Z crate-attr`.

use rustc_ast::{self as ast};
use rustc_errors::Diag;
use rustc_parse::parser::attr::InnerAttrPolicy;
use rustc_parse::{parse_in, source_str_to_stream};
use rustc_session::parse::ParseSess;
use rustc_span::FileName;

pub fn inject(krate: &mut ast::Crate, psess: &ParseSess, attrs: &[String]) {
    for raw_attr in attrs {
        let source = format!("#![{raw_attr}]");
        let parse = || -> Result<ast::Attribute, Vec<Diag<'_>>> {
            let tokens = source_str_to_stream(
                psess,
                FileName::cli_crate_attr_source_code(raw_attr),
                source,
                None,
            )?;
            parse_in(psess, tokens, "<crate attribute>", |p| {
                p.parse_attribute(InnerAttrPolicy::Permitted)
            })
            .map_err(|e| vec![e])
        };
        let meta = match parse() {
            Ok(meta) => meta,
            Err(errs) => {
                for err in errs {
                    err.emit();
                }
                continue;
            }
        };

        krate.attrs.push(meta);
    }
}
