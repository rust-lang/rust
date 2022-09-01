use super::*;
use rustc_ast as ast;
use rustc_span::sym;

pub fn expand_parsed_format_args(ecx: &mut ExtCtxt<'_>, fmt: FormatArgs) -> P<ast::Expr> {
    let macsp = ecx.with_def_site_ctxt(ecx.call_site());

    … // TODO

    // Generate:
    //     ::core::fmt::Arguments::new(
    //         …
    //     )
    ecx.expr_call_global(
        macsp,
        ecx.std_path(&[sym::fmt, sym::Arguments, sym::new]),
        vec![…],
    )
}
