use rustc_ast as ast;
use rustc_ast::ptr::P;
use rustc_ast::token;
use rustc_ast::tokenstream::TokenStream;
use rustc_ast_pretty::pprust;
use rustc_expand::base::{
    check_zero_tts, get_single_str_from_tts, parse_expr, resolve_path, DummyResult, ExtCtxt,
    MacEager, MacResult,
};
use rustc_expand::module::DirOwnership;
use rustc_parse::new_parser_from_file;
use rustc_parse::parser::{ForceCollect, Parser};
use rustc_session::lint::builtin::INCOMPLETE_INCLUDE;
use rustc_span::symbol::Symbol;
use rustc_span::{Pos, Span};

use smallvec::SmallVec;
use std::rc::Rc;

// These macros all relate to the file system; they either return
// the column/row/filename of the expression, or they include
// a given file into the current one.

/// line!(): expands to the current line number
pub fn expand_line(
    cx: &mut ExtCtxt<'_>,
    sp: Span,
    tts: TokenStream,
) -> Box<dyn MacResult + 'static> {
    let sp = cx.with_def_site_ctxt(sp);
    check_zero_tts(cx, sp, tts, "line!");

    let topmost = cx.expansion_cause().unwrap_or(sp);
    let loc = cx.source_map().lookup_char_pos(topmost.lo());

    MacEager::expr(cx.expr_u32(topmost, loc.line as u32))
}

/* column!(): expands to the current column number */
pub fn expand_column(
    cx: &mut ExtCtxt<'_>,
    sp: Span,
    tts: TokenStream,
) -> Box<dyn MacResult + 'static> {
    let sp = cx.with_def_site_ctxt(sp);
    check_zero_tts(cx, sp, tts, "column!");

    let topmost = cx.expansion_cause().unwrap_or(sp);
    let loc = cx.source_map().lookup_char_pos(topmost.lo());

    MacEager::expr(cx.expr_u32(topmost, loc.col.to_usize() as u32 + 1))
}

/// file!(): expands to the current filename */
/// The source_file (`loc.file`) contains a bunch more information we could spit
/// out if we wanted.
pub fn expand_file(
    cx: &mut ExtCtxt<'_>,
    sp: Span,
    tts: TokenStream,
) -> Box<dyn MacResult + 'static> {
    let sp = cx.with_def_site_ctxt(sp);
    check_zero_tts(cx, sp, tts, "file!");

    let topmost = cx.expansion_cause().unwrap_or(sp);
    let loc = cx.source_map().lookup_char_pos(topmost.lo());

    use rustc_session::{config::RemapPathScopeComponents, RemapFileNameExt};
    MacEager::expr(cx.expr_str(
        topmost,
        Symbol::intern(
            &loc.file.name.for_scope(cx.sess, RemapPathScopeComponents::MACRO).to_string_lossy(),
        ),
    ))
}

pub fn expand_stringify(
    cx: &mut ExtCtxt<'_>,
    sp: Span,
    tts: TokenStream,
) -> Box<dyn MacResult + 'static> {
    let sp = cx.with_def_site_ctxt(sp);
    let s = pprust::tts_to_string(&tts);
    MacEager::expr(cx.expr_str(sp, Symbol::intern(&s)))
}

pub fn expand_mod(
    cx: &mut ExtCtxt<'_>,
    sp: Span,
    tts: TokenStream,
) -> Box<dyn MacResult + 'static> {
    let sp = cx.with_def_site_ctxt(sp);
    check_zero_tts(cx, sp, tts, "module_path!");
    let mod_path = &cx.current_expansion.module.mod_path;
    let string = mod_path.iter().map(|x| x.to_string()).collect::<Vec<String>>().join("::");

    MacEager::expr(cx.expr_str(sp, Symbol::intern(&string)))
}

/// include! : parse the given file as an expr
/// This is generally a bad idea because it's going to behave
/// unhygienically.
pub fn expand_include<'cx>(
    cx: &'cx mut ExtCtxt<'_>,
    sp: Span,
    tts: TokenStream,
) -> Box<dyn MacResult + 'cx> {
    let sp = cx.with_def_site_ctxt(sp);
    let file = match get_single_str_from_tts(cx, sp, tts, "include!") {
        Ok(file) => file,
        Err(guar) => return DummyResult::any(sp, guar),
    };
    // The file will be added to the code map by the parser
    let file = match resolve_path(&cx.sess, file.as_str(), sp) {
        Ok(f) => f,
        Err(err) => {
            let guar = err.emit();
            return DummyResult::any(sp, guar);
        }
    };
    let p = new_parser_from_file(cx.psess(), &file, Some(sp));

    // If in the included file we have e.g., `mod bar;`,
    // then the path of `bar.rs` should be relative to the directory of `file`.
    // See https://github.com/rust-lang/rust/pull/69838/files#r395217057 for a discussion.
    // `MacroExpander::fully_expand_fragment` later restores, so "stack discipline" is maintained.
    let dir_path = file.parent().unwrap_or(&file).to_owned();
    cx.current_expansion.module = Rc::new(cx.current_expansion.module.with_dir_path(dir_path));
    cx.current_expansion.dir_ownership = DirOwnership::Owned { relative: None };

    struct ExpandResult<'a> {
        p: Parser<'a>,
        node_id: ast::NodeId,
    }
    impl<'a> MacResult for ExpandResult<'a> {
        fn make_expr(mut self: Box<ExpandResult<'a>>) -> Option<P<ast::Expr>> {
            let expr = parse_expr(&mut self.p).ok()?;
            if self.p.token != token::Eof {
                self.p.psess.buffer_lint(
                    INCOMPLETE_INCLUDE,
                    self.p.token.span,
                    self.node_id,
                    "include macro expected single expression in source",
                );
            }
            Some(expr)
        }

        fn make_items(mut self: Box<ExpandResult<'a>>) -> Option<SmallVec<[P<ast::Item>; 1]>> {
            let mut ret = SmallVec::new();
            loop {
                match self.p.parse_item(ForceCollect::No) {
                    Err(err) => {
                        err.emit();
                        break;
                    }
                    Ok(Some(item)) => ret.push(item),
                    Ok(None) => {
                        if self.p.token != token::Eof {
                            let token = pprust::token_to_string(&self.p.token);
                            let msg = format!("expected item, found `{token}`");
                            self.p.dcx().span_err(self.p.token.span, msg);
                        }

                        break;
                    }
                }
            }
            Some(ret)
        }
    }

    Box::new(ExpandResult { p, node_id: cx.current_expansion.lint_node_id })
}

/// `include_str!`: read the given file, insert it as a literal string expr
pub fn expand_include_str(
    cx: &mut ExtCtxt<'_>,
    sp: Span,
    tts: TokenStream,
) -> Box<dyn MacResult + 'static> {
    let sp = cx.with_def_site_ctxt(sp);
    let file = match get_single_str_from_tts(cx, sp, tts, "include_str!") {
        Ok(file) => file,
        Err(guar) => return DummyResult::any(sp, guar),
    };
    let file = match resolve_path(&cx.sess, file.as_str(), sp) {
        Ok(f) => f,
        Err(err) => {
            let guar = err.emit();
            return DummyResult::any(sp, guar);
        }
    };
    match cx.source_map().load_binary_file(&file) {
        Ok(bytes) => match std::str::from_utf8(&bytes) {
            Ok(src) => {
                let interned_src = Symbol::intern(src);
                MacEager::expr(cx.expr_str(sp, interned_src))
            }
            Err(_) => {
                let guar = cx.dcx().span_err(sp, format!("{} wasn't a utf-8 file", file.display()));
                DummyResult::any(sp, guar)
            }
        },
        Err(e) => {
            let guar = cx.dcx().span_err(sp, format!("couldn't read {}: {}", file.display(), e));
            DummyResult::any(sp, guar)
        }
    }
}

pub fn expand_include_bytes(
    cx: &mut ExtCtxt<'_>,
    sp: Span,
    tts: TokenStream,
) -> Box<dyn MacResult + 'static> {
    let sp = cx.with_def_site_ctxt(sp);
    let file = match get_single_str_from_tts(cx, sp, tts, "include_bytes!") {
        Ok(file) => file,
        Err(guar) => return DummyResult::any(sp, guar),
    };
    let file = match resolve_path(&cx.sess, file.as_str(), sp) {
        Ok(f) => f,
        Err(err) => {
            let guar = err.emit();
            return DummyResult::any(sp, guar);
        }
    };
    match cx.source_map().load_binary_file(&file) {
        Ok(bytes) => {
            let expr = cx.expr(sp, ast::ExprKind::IncludedBytes(bytes));
            MacEager::expr(expr)
        }
        Err(e) => {
            let guar = cx.dcx().span_err(sp, format!("couldn't read {}: {}", file.display(), e));
            DummyResult::any(sp, guar)
        }
    }
}
