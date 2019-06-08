use crate::ast;
use crate::ext::base::{self, *};
use crate::ext::build::AstBuilder;
use crate::parse::{self, token, DirectoryOwnership};
use crate::print::pprust;
use crate::ptr::P;
use crate::symbol::{Symbol, sym};
use crate::tokenstream;

use smallvec::SmallVec;
use syntax_pos::{self, Pos, Span, FileName};

use std::fs;
use std::io::ErrorKind;
use std::path::PathBuf;
use rustc_data_structures::sync::Lrc;

// These macros all relate to the file system; they either return
// the column/row/filename of the expression, or they include
// a given file into the current one.

/// line!(): expands to the current line number
pub fn expand_line(cx: &mut ExtCtxt<'_>, sp: Span, tts: &[tokenstream::TokenTree])
                   -> Box<dyn base::MacResult+'static> {
    base::check_zero_tts(cx, sp, tts, "line!");

    let topmost = cx.expansion_cause().unwrap_or(sp);
    let loc = cx.source_map().lookup_char_pos(topmost.lo());

    base::MacEager::expr(cx.expr_u32(topmost, loc.line as u32))
}

/* column!(): expands to the current column number */
pub fn expand_column(cx: &mut ExtCtxt<'_>, sp: Span, tts: &[tokenstream::TokenTree])
                  -> Box<dyn base::MacResult+'static> {
    base::check_zero_tts(cx, sp, tts, "column!");

    let topmost = cx.expansion_cause().unwrap_or(sp);
    let loc = cx.source_map().lookup_char_pos(topmost.lo());

    base::MacEager::expr(cx.expr_u32(topmost, loc.col.to_usize() as u32 + 1))
}

/* __rust_unstable_column!(): expands to the current column number */
pub fn expand_column_gated(cx: &mut ExtCtxt<'_>, sp: Span, tts: &[tokenstream::TokenTree])
                  -> Box<dyn base::MacResult+'static> {
    if sp.allows_unstable(sym::__rust_unstable_column) {
        expand_column(cx, sp, tts)
    } else {
        cx.span_fatal(sp, "the __rust_unstable_column macro is unstable");
    }
}

/// file!(): expands to the current filename */
/// The source_file (`loc.file`) contains a bunch more information we could spit
/// out if we wanted.
pub fn expand_file(cx: &mut ExtCtxt<'_>, sp: Span, tts: &[tokenstream::TokenTree])
                   -> Box<dyn base::MacResult+'static> {
    base::check_zero_tts(cx, sp, tts, "file!");

    let topmost = cx.expansion_cause().unwrap_or(sp);
    let loc = cx.source_map().lookup_char_pos(topmost.lo());
    base::MacEager::expr(cx.expr_str(topmost, Symbol::intern(&loc.file.name.to_string())))
}

pub fn expand_stringify(cx: &mut ExtCtxt<'_>, sp: Span, tts: &[tokenstream::TokenTree])
                        -> Box<dyn base::MacResult+'static> {
    let s = pprust::tts_to_string(tts);
    base::MacEager::expr(cx.expr_str(sp, Symbol::intern(&s)))
}

pub fn expand_mod(cx: &mut ExtCtxt<'_>, sp: Span, tts: &[tokenstream::TokenTree])
                  -> Box<dyn base::MacResult+'static> {
    base::check_zero_tts(cx, sp, tts, "module_path!");
    let mod_path = &cx.current_expansion.module.mod_path;
    let string = mod_path.iter().map(|x| x.to_string()).collect::<Vec<String>>().join("::");

    base::MacEager::expr(cx.expr_str(sp, Symbol::intern(&string)))
}

/// include! : parse the given file as an expr
/// This is generally a bad idea because it's going to behave
/// unhygienically.
pub fn expand_include<'cx>(cx: &'cx mut ExtCtxt<'_>, sp: Span, tts: &[tokenstream::TokenTree])
                           -> Box<dyn base::MacResult+'cx> {
    let file = match get_single_str_from_tts(cx, sp, tts, "include!") {
        Some(f) => f,
        None => return DummyResult::any(sp),
    };
    // The file will be added to the code map by the parser
    let path = res_rel_file(cx, sp, file);
    let directory_ownership = DirectoryOwnership::Owned { relative: None };
    let p = parse::new_sub_parser_from_file(cx.parse_sess(), &path, directory_ownership, None, sp);

    struct ExpandResult<'a> {
        p: parse::parser::Parser<'a>,
    }
    impl<'a> base::MacResult for ExpandResult<'a> {
        fn make_expr(mut self: Box<ExpandResult<'a>>) -> Option<P<ast::Expr>> {
            Some(panictry!(self.p.parse_expr()))
        }

        fn make_items(mut self: Box<ExpandResult<'a>>) -> Option<SmallVec<[P<ast::Item>; 1]>> {
            let mut ret = SmallVec::new();
            while self.p.token != token::Eof {
                match panictry!(self.p.parse_item()) {
                    Some(item) => ret.push(item),
                    None => self.p.diagnostic().span_fatal(self.p.token.span,
                                                           &format!("expected item, found `{}`",
                                                                    self.p.this_token_to_string()))
                                               .raise()
                }
            }
            Some(ret)
        }
    }

    Box::new(ExpandResult { p })
}

// include_str! : read the given file, insert it as a literal string expr
pub fn expand_include_str(cx: &mut ExtCtxt<'_>, sp: Span, tts: &[tokenstream::TokenTree])
                          -> Box<dyn base::MacResult+'static> {
    let file = match get_single_str_from_tts(cx, sp, tts, "include_str!") {
        Some(f) => f,
        None => return DummyResult::expr(sp)
    };
    let file = res_rel_file(cx, sp, file);
    match fs::read_to_string(&file) {
        Ok(src) => {
            let interned_src = Symbol::intern(&src);

            // Add this input file to the code map to make it available as
            // dependency information
            cx.source_map().new_source_file(file.into(), src);

            base::MacEager::expr(cx.expr_str(sp, interned_src))
        },
        Err(ref e) if e.kind() == ErrorKind::InvalidData => {
            cx.span_err(sp, &format!("{} wasn't a utf-8 file", file.display()));
            DummyResult::expr(sp)
        }
        Err(e) => {
            cx.span_err(sp, &format!("couldn't read {}: {}", file.display(), e));
            DummyResult::expr(sp)
        }
    }
}

pub fn expand_include_bytes(cx: &mut ExtCtxt<'_>, sp: Span, tts: &[tokenstream::TokenTree])
                            -> Box<dyn base::MacResult+'static> {
    let file = match get_single_str_from_tts(cx, sp, tts, "include_bytes!") {
        Some(f) => f,
        None => return DummyResult::expr(sp)
    };
    let file = res_rel_file(cx, sp, file);
    match fs::read(&file) {
        Ok(bytes) => {
            // Add the contents to the source map if it contains UTF-8.
            let (contents, bytes) = match String::from_utf8(bytes) {
                Ok(s) => {
                    let bytes = s.as_bytes().to_owned();
                    (s, bytes)
                },
                Err(e) => (String::new(), e.into_bytes()),
            };
            cx.source_map().new_source_file(file.into(), contents);

            base::MacEager::expr(cx.expr_lit(sp, ast::LitKind::ByteStr(Lrc::new(bytes))))
        },
        Err(e) => {
            cx.span_err(sp, &format!("couldn't read {}: {}", file.display(), e));
            DummyResult::expr(sp)
        }
    }
}

// resolve a file-system path to an absolute file-system path (if it
// isn't already)
fn res_rel_file(cx: &mut ExtCtxt<'_>, sp: syntax_pos::Span, arg: String) -> PathBuf {
    let arg = PathBuf::from(arg);
    // Relative paths are resolved relative to the file in which they are found
    // after macro expansion (that is, they are unhygienic).
    if !arg.is_absolute() {
        let callsite = sp.source_callsite();
        let mut path = match cx.source_map().span_to_unmapped_path(callsite) {
            FileName::Real(path) => path,
            FileName::DocTest(path, _) => path,
            other => panic!("cannot resolve relative path in non-file source `{}`", other),
        };
        path.pop();
        path.push(arg);
        path
    } else {
        arg
    }
}
