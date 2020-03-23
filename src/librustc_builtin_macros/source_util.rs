use rustc_ast::ast;
use rustc_ast::ptr::P;
use rustc_ast::token;
use rustc_ast::tokenstream::TokenStream;
use rustc_ast_pretty::pprust;
use rustc_expand::base::{self, *};
use rustc_expand::module::DirectoryOwnership;
use rustc_expand::panictry;
use rustc_parse::{self, new_parser_from_file, parser::Parser};
use rustc_session::lint::builtin::INCOMPLETE_INCLUDE;
use rustc_span::symbol::Symbol;
use rustc_span::{self, Pos, Span};

use smallvec::SmallVec;
use std::rc::Rc;

use rustc_data_structures::sync::Lrc;

// These macros all relate to the file system; they either return
// the column/row/filename of the expression, or they include
// a given file into the current one.

/// line!(): expands to the current line number
pub fn expand_line(
    cx: &mut ExtCtxt<'_>,
    sp: Span,
    tts: TokenStream,
) -> Box<dyn base::MacResult + 'static> {
    let sp = cx.with_def_site_ctxt(sp);
    base::check_zero_tts(cx, sp, tts, "line!");

    let topmost = cx.expansion_cause().unwrap_or(sp);
    let loc = cx.source_map().lookup_char_pos(topmost.lo());

    base::MacEager::expr(cx.expr_u32(topmost, loc.line as u32))
}

/* column!(): expands to the current column number */
pub fn expand_column(
    cx: &mut ExtCtxt<'_>,
    sp: Span,
    tts: TokenStream,
) -> Box<dyn base::MacResult + 'static> {
    let sp = cx.with_def_site_ctxt(sp);
    base::check_zero_tts(cx, sp, tts, "column!");

    let topmost = cx.expansion_cause().unwrap_or(sp);
    let loc = cx.source_map().lookup_char_pos(topmost.lo());

    base::MacEager::expr(cx.expr_u32(topmost, loc.col.to_usize() as u32 + 1))
}

/// file!(): expands to the current filename */
/// The source_file (`loc.file`) contains a bunch more information we could spit
/// out if we wanted.
pub fn expand_file(
    cx: &mut ExtCtxt<'_>,
    sp: Span,
    tts: TokenStream,
) -> Box<dyn base::MacResult + 'static> {
    let sp = cx.with_def_site_ctxt(sp);
    base::check_zero_tts(cx, sp, tts, "file!");

    let topmost = cx.expansion_cause().unwrap_or(sp);
    let loc = cx.source_map().lookup_char_pos(topmost.lo());
    base::MacEager::expr(cx.expr_str(topmost, Symbol::intern(&loc.file.name.to_string())))
}

pub fn expand_stringify(
    cx: &mut ExtCtxt<'_>,
    sp: Span,
    tts: TokenStream,
) -> Box<dyn base::MacResult + 'static> {
    let sp = cx.with_def_site_ctxt(sp);
    let s = pprust::tts_to_string(tts);
    base::MacEager::expr(cx.expr_str(sp, Symbol::intern(&s)))
}

pub fn expand_mod(
    cx: &mut ExtCtxt<'_>,
    sp: Span,
    tts: TokenStream,
) -> Box<dyn base::MacResult + 'static> {
    let sp = cx.with_def_site_ctxt(sp);
    base::check_zero_tts(cx, sp, tts, "module_path!");
    let mod_path = &cx.current_expansion.module.mod_path;
    let string = mod_path.iter().map(|x| x.to_string()).collect::<Vec<String>>().join("::");

    base::MacEager::expr(cx.expr_str(sp, Symbol::intern(&string)))
}

/// include! : parse the given file as an expr
/// This is generally a bad idea because it's going to behave
/// unhygienically.
pub fn expand_include<'cx>(
    cx: &'cx mut ExtCtxt<'_>,
    sp: Span,
    tts: TokenStream,
) -> Box<dyn base::MacResult + 'cx> {
    let sp = cx.with_def_site_ctxt(sp);
    let file = match get_single_str_from_tts(cx, sp, tts, "include!") {
        Some(f) => f,
        None => return DummyResult::any(sp),
    };
    // The file will be added to the code map by the parser
    let mut file = match cx.resolve_path(file, sp) {
        Ok(f) => f,
        Err(mut err) => {
            err.emit();
            return DummyResult::any(sp);
        }
    };
    let p = new_parser_from_file(cx.parse_sess(), &file, Some(sp));

    // If in the included file we have e.g., `mod bar;`,
    // then the path of `bar.rs` should be relative to the directory of `file`.
    // See https://github.com/rust-lang/rust/pull/69838/files#r395217057 for a discussion.
    // `MacroExpander::fully_expand_fragment` later restores, so "stack discipline" is maintained.
    file.pop();
    cx.current_expansion.directory_ownership = DirectoryOwnership::Owned { relative: None };
    let mod_path = cx.current_expansion.module.mod_path.clone();
    cx.current_expansion.module = Rc::new(ModuleData { mod_path, directory: file });

    struct ExpandResult<'a> {
        p: Parser<'a>,
    }
    impl<'a> base::MacResult for ExpandResult<'a> {
        fn make_expr(mut self: Box<ExpandResult<'a>>) -> Option<P<ast::Expr>> {
            let r = panictry!(self.p.parse_expr());
            if self.p.token != token::Eof {
                self.p.sess.buffer_lint(
                    &INCOMPLETE_INCLUDE,
                    self.p.token.span,
                    ast::CRATE_NODE_ID,
                    "include macro expected single expression in source",
                );
            }
            Some(r)
        }

        fn make_items(mut self: Box<ExpandResult<'a>>) -> Option<SmallVec<[P<ast::Item>; 1]>> {
            let mut ret = SmallVec::new();
            while self.p.token != token::Eof {
                match panictry!(self.p.parse_item()) {
                    Some(item) => ret.push(item),
                    None => {
                        let token = pprust::token_to_string(&self.p.token);
                        self.p
                            .sess
                            .span_diagnostic
                            .span_fatal(
                                self.p.token.span,
                                &format!("expected item, found `{}`", token),
                            )
                            .raise();
                    }
                }
            }
            Some(ret)
        }
    }

    Box::new(ExpandResult { p })
}

// include_str! : read the given file, insert it as a literal string expr
pub fn expand_include_str(
    cx: &mut ExtCtxt<'_>,
    sp: Span,
    tts: TokenStream,
) -> Box<dyn base::MacResult + 'static> {
    let sp = cx.with_def_site_ctxt(sp);
    let file = match get_single_str_from_tts(cx, sp, tts, "include_str!") {
        Some(f) => f,
        None => return DummyResult::any(sp),
    };
    let file = match cx.resolve_path(file, sp) {
        Ok(f) => f,
        Err(mut err) => {
            err.emit();
            return DummyResult::any(sp);
        }
    };
    match cx.source_map().load_binary_file(&file) {
        Ok(bytes) => match std::str::from_utf8(&bytes) {
            Ok(src) => {
                let interned_src = Symbol::intern(&src);
                base::MacEager::expr(cx.expr_str(sp, interned_src))
            }
            Err(_) => {
                cx.span_err(sp, &format!("{} wasn't a utf-8 file", file.display()));
                DummyResult::any(sp)
            }
        },
        Err(e) => {
            cx.span_err(sp, &format!("couldn't read {}: {}", file.display(), e));
            DummyResult::any(sp)
        }
    }
}

pub fn expand_include_bytes(
    cx: &mut ExtCtxt<'_>,
    sp: Span,
    tts: TokenStream,
) -> Box<dyn base::MacResult + 'static> {
    let sp = cx.with_def_site_ctxt(sp);
    let file = match get_single_str_from_tts(cx, sp, tts, "include_bytes!") {
        Some(f) => f,
        None => return DummyResult::any(sp),
    };
    let file = match cx.resolve_path(file, sp) {
        Ok(f) => f,
        Err(mut err) => {
            err.emit();
            return DummyResult::any(sp);
        }
    };
    match cx.source_map().load_binary_file(&file) {
        Ok(bytes) => base::MacEager::expr(cx.expr_lit(sp, ast::LitKind::ByteStr(Lrc::new(bytes)))),
        Err(e) => {
            cx.span_err(sp, &format!("couldn't read {}: {}", file.display(), e));
            DummyResult::any(sp)
        }
    }
}
