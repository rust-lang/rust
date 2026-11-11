//! Attributes injected into the crate root from command line using `-Z crate-attr`.

use rustc_ast as ast;
use rustc_errors::Diag;
use rustc_parse::parser::attr::InnerAttrPolicy;
use rustc_parse::{parse_in, source_str_to_stream};
use rustc_session::parse::ParseSess;
use rustc_span::FileName;

pub fn inject(krate: &mut ast::Crate, psess: &ParseSess, attrs: &[String]) {
    let injected = attrs.iter().filter_map(|raw_attr| {
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
        match parse() {
            Ok(meta) => Some(meta),
            Err(errs) => {
                for err in errs {
                    err.emit();
                }
                None
            }
        }
    });
    krate.attrs.splice(0..0, injected);
}
