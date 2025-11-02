//! The implementation of built-in macros which relate to the file system.

use std::path::{Path, PathBuf};
use std::rc::Rc;
use std::sync::Arc;

use rustc_ast as ast;
use rustc_ast::tokenstream::TokenStream;
use rustc_ast::{join_path_idents, token};
use rustc_ast_pretty::pprust;
use rustc_expand::base::{
    DummyResult, ExpandResult, ExtCtxt, MacEager, MacResult, MacroExpanderResult, resolve_path,
};
use rustc_expand::module::DirOwnership;
use rustc_parse::lexer::StripTokens;
use rustc_parse::parser::ForceCollect;
use rustc_parse::{new_parser_from_file, unwrap_or_emit_fatal, utf8_error};
use rustc_session::lint::builtin::INCOMPLETE_INCLUDE;
use rustc_session::parse::ParseSess;
use rustc_span::source_map::SourceMap;
use rustc_span::{ByteSymbol, Pos, Span, Symbol};
use smallvec::SmallVec;

use crate::errors;
use crate::util::{
    check_zero_tts, get_single_str_from_tts, get_single_str_spanned_from_tts, parse_expr,
};

/// Expand `line!()` to the current line number.
pub(crate) fn expand_line(
    cx: &mut ExtCtxt<'_>,
    sp: Span,
    tts: TokenStream,
) -> MacroExpanderResult<'static> {
    let sp = cx.with_def_site_ctxt(sp);
    check_zero_tts(cx, sp, tts, "line!");

    let topmost = cx.expansion_cause().unwrap_or(sp);
    let loc = cx.source_map().lookup_char_pos(topmost.lo());

    ExpandResult::Ready(MacEager::expr(cx.expr_u32(topmost, loc.line as u32)))
}

/// Expand `column!()` to the current column number.
pub(crate) fn expand_column(
    cx: &mut ExtCtxt<'_>,
    sp: Span,
    tts: TokenStream,
) -> MacroExpanderResult<'static> {
    let sp = cx.with_def_site_ctxt(sp);
    check_zero_tts(cx, sp, tts, "column!");

    let topmost = cx.expansion_cause().unwrap_or(sp);
    let loc = cx.source_map().lookup_char_pos(topmost.lo());

    ExpandResult::Ready(MacEager::expr(cx.expr_u32(topmost, loc.col.to_usize() as u32 + 1)))
}

/// Expand `file!()` to the current filename.
pub(crate) fn expand_file(
    cx: &mut ExtCtxt<'_>,
    sp: Span,
    tts: TokenStream,
) -> MacroExpanderResult<'static> {
    let sp = cx.with_def_site_ctxt(sp);
    check_zero_tts(cx, sp, tts, "file!");

    let topmost = cx.expansion_cause().unwrap_or(sp);
    let loc = cx.source_map().lookup_char_pos(topmost.lo());

    use rustc_session::RemapFileNameExt;
    use rustc_session::config::RemapPathScopeComponents;
    ExpandResult::Ready(MacEager::expr(cx.expr_str(
        topmost,
        Symbol::intern(
            &loc.file.name.for_scope(cx.sess, RemapPathScopeComponents::MACRO).to_string_lossy(),
        ),
    )))
}

/// Expand `stringify!($input)`.
pub(crate) fn expand_stringify(
    cx: &mut ExtCtxt<'_>,
    sp: Span,
    tts: TokenStream,
) -> MacroExpanderResult<'static> {
    let sp = cx.with_def_site_ctxt(sp);
    let s = pprust::tts_to_string(&tts);
    ExpandResult::Ready(MacEager::expr(cx.expr_str(sp, Symbol::intern(&s))))
}

/// Expand `module_path!()` to (a textual representation of) the current module path.
pub(crate) fn expand_mod(
    cx: &mut ExtCtxt<'_>,
    sp: Span,
    tts: TokenStream,
) -> MacroExpanderResult<'static> {
    let sp = cx.with_def_site_ctxt(sp);
    check_zero_tts(cx, sp, tts, "module_path!");
    let mod_path = &cx.current_expansion.module.mod_path;
    let string = join_path_idents(mod_path);

    ExpandResult::Ready(MacEager::expr(cx.expr_str(sp, Symbol::intern(&string))))
}

/// Expand `include!($input)`.
///
/// This works in item and expression position. Notably, it doesn't work in pattern position.
pub(crate) fn expand_include<'cx>(
    cx: &'cx mut ExtCtxt<'_>,
    sp: Span,
    tts: TokenStream,
) -> MacroExpanderResult<'cx> {
    let sp = cx.with_def_site_ctxt(sp);
    let ExpandResult::Ready(mac) = get_single_str_from_tts(cx, sp, tts, "include!") else {
        return ExpandResult::Retry(());
    };
    let path = match mac {
        Ok(path) => path,
        Err(guar) => return ExpandResult::Ready(DummyResult::any(sp, guar)),
    };
    // The file will be added to the code map by the parser
    let path = match resolve_path(&cx.sess, path.as_str(), sp) {
        Ok(path) => path,
        Err(err) => {
            let guar = err.emit();
            return ExpandResult::Ready(DummyResult::any(sp, guar));
        }
    };

    // If in the included file we have e.g., `mod bar;`,
    // then the path of `bar.rs` should be relative to the directory of `path`.
    // See https://github.com/rust-lang/rust/pull/69838/files#r395217057 for a discussion.
    // `MacroExpander::fully_expand_fragment` later restores, so "stack discipline" is maintained.
    let dir_path = path.parent().unwrap_or(&path).to_owned();
    cx.current_expansion.module = Rc::new(cx.current_expansion.module.with_dir_path(dir_path));
    cx.current_expansion.dir_ownership = DirOwnership::Owned { relative: None };

    struct ExpandInclude<'a> {
        psess: &'a ParseSess,
        path: PathBuf,
        node_id: ast::NodeId,
        span: Span,
    }
    impl<'a> MacResult for ExpandInclude<'a> {
        fn make_expr(self: Box<ExpandInclude<'a>>) -> Option<Box<ast::Expr>> {
            let mut p = unwrap_or_emit_fatal(new_parser_from_file(
                self.psess,
                &self.path,
                // Don't strip frontmatter for backward compatibility, `---` may be the start of a
                // manifold negation. FIXME: Ideally, we wouldn't strip shebangs here either.
                StripTokens::Shebang,
                Some(self.span),
            ));
            let expr = parse_expr(&mut p).ok()?;
            if p.token != token::Eof {
                p.psess.buffer_lint(
                    INCOMPLETE_INCLUDE,
                    p.token.span,
                    self.node_id,
                    errors::IncompleteInclude,
                );
            }
            Some(expr)
        }

        fn make_items(self: Box<ExpandInclude<'a>>) -> Option<SmallVec<[Box<ast::Item>; 1]>> {
            let mut p = unwrap_or_emit_fatal(new_parser_from_file(
                self.psess,
                &self.path,
                StripTokens::ShebangAndFrontmatter,
                Some(self.span),
            ));
            let mut ret = SmallVec::new();
            loop {
                match p.parse_item(ForceCollect::No) {
                    Err(err) => {
                        err.emit();
                        break;
                    }
                    Ok(Some(item)) => ret.push(item),
                    Ok(None) => {
                        if p.token != token::Eof {
                            p.dcx().emit_err(errors::ExpectedItem {
                                span: p.token.span,
                                token: &pprust::token_to_string(&p.token),
                            });
                        }

                        break;
                    }
                }
            }
            Some(ret)
        }
    }

    ExpandResult::Ready(Box::new(ExpandInclude {
        psess: cx.psess(),
        path,
        node_id: cx.current_expansion.lint_node_id,
        span: sp,
    }))
}

/// Expand `include_str!($input)` to the content of the UTF-8-encoded file given by path `$input` as a string literal.
///
/// This works in expression, pattern and statement position.
pub(crate) fn expand_include_str(
    cx: &mut ExtCtxt<'_>,
    sp: Span,
    tts: TokenStream,
) -> MacroExpanderResult<'static> {
    let sp = cx.with_def_site_ctxt(sp);
    let ExpandResult::Ready(mac) = get_single_str_spanned_from_tts(cx, sp, tts, "include_str!")
    else {
        return ExpandResult::Retry(());
    };
    let (path, path_span) = match mac {
        Ok(res) => res,
        Err(guar) => return ExpandResult::Ready(DummyResult::any(sp, guar)),
    };
    ExpandResult::Ready(match load_binary_file(cx, path.as_str().as_ref(), sp, path_span) {
        Ok((bytes, bsp)) => match std::str::from_utf8(&bytes) {
            Ok(src) => {
                let interned_src = Symbol::intern(src);
                // MacEager converts the expr into a pat if need be.
                MacEager::expr(cx.expr_str(cx.with_def_site_ctxt(bsp), interned_src))
            }
            Err(utf8err) => {
                let mut err = cx.dcx().struct_span_err(sp, format!("`{path}` wasn't a utf-8 file"));
                utf8_error(cx.source_map(), path.as_str(), None, &mut err, utf8err, &bytes[..]);
                DummyResult::any(sp, err.emit())
            }
        },
        Err(dummy) => dummy,
    })
}

/// Expand `include_bytes!($input)` to the content of the file given by path `$input`.
///
/// This works in expression, pattern and statement position.
pub(crate) fn expand_include_bytes(
    cx: &mut ExtCtxt<'_>,
    sp: Span,
    tts: TokenStream,
) -> MacroExpanderResult<'static> {
    let sp = cx.with_def_site_ctxt(sp);
    let ExpandResult::Ready(mac) = get_single_str_spanned_from_tts(cx, sp, tts, "include_bytes!")
    else {
        return ExpandResult::Retry(());
    };
    let (path, path_span) = match mac {
        Ok(res) => res,
        Err(guar) => return ExpandResult::Ready(DummyResult::any(sp, guar)),
    };
    ExpandResult::Ready(match load_binary_file(cx, path.as_str().as_ref(), sp, path_span) {
        Ok((bytes, _bsp)) => {
            // Don't care about getting the span for the raw bytes,
            // because the console can't really show them anyway.
            let expr = cx.expr(sp, ast::ExprKind::IncludedBytes(ByteSymbol::intern(&bytes)));
            // MacEager converts the expr into a pat if need be.
            MacEager::expr(expr)
        }
        Err(dummy) => dummy,
    })
}

fn load_binary_file(
    cx: &ExtCtxt<'_>,
    original_path: &Path,
    macro_span: Span,
    path_span: Span,
) -> Result<(Arc<[u8]>, Span), Box<dyn MacResult>> {
    let resolved_path = match resolve_path(&cx.sess, original_path, macro_span) {
        Ok(path) => path,
        Err(err) => {
            let guar = err.emit();
            return Err(DummyResult::any(macro_span, guar));
        }
    };
    match cx.source_map().load_binary_file(&resolved_path) {
        Ok(data) => Ok(data),
        Err(io_err) => {
            let mut err = cx.dcx().struct_span_err(
                macro_span,
                format!("couldn't read `{}`: {io_err}", resolved_path.display()),
            );

            if original_path.is_relative() {
                let source_map = cx.sess.source_map();
                let new_path = source_map
                    .span_to_filename(macro_span.source_callsite())
                    .into_local_path()
                    .and_then(|src| find_path_suggestion(source_map, src.parent()?, original_path))
                    .and_then(|path| path.into_os_string().into_string().ok());

                if let Some(new_path) = new_path {
                    err.span_suggestion_verbose(
                        path_span,
                        "there is a file with the same name in a different directory",
                        format!("\"{}\"", new_path.replace('\\', "/").escape_debug()),
                        rustc_lint_defs::Applicability::MachineApplicable,
                    );
                }
            }
            let guar = err.emit();
            Err(DummyResult::any(macro_span, guar))
        }
    }
}

fn find_path_suggestion(
    source_map: &SourceMap,
    base_dir: &Path,
    wanted_path: &Path,
) -> Option<PathBuf> {
    // Fix paths that assume they're relative to cargo manifest dir
    let mut base_c = base_dir.components();
    let mut wanted_c = wanted_path.components();
    let mut without_base = None;
    while let Some(wanted_next) = wanted_c.next() {
        if wanted_c.as_path().file_name().is_none() {
            break;
        }
        // base_dir may be absolute
        while let Some(base_next) = base_c.next() {
            if base_next == wanted_next {
                without_base = Some(wanted_c.as_path());
                break;
            }
        }
    }
    let root_absolute = without_base.into_iter().map(PathBuf::from);

    let base_dir_components = base_dir.components().count();
    // Avoid going all the way to the root dir
    let max_parent_components = if base_dir.is_relative() {
        base_dir_components + 1
    } else {
        base_dir_components.saturating_sub(1)
    };

    // Try with additional leading ../
    let mut prefix = PathBuf::new();
    let add = std::iter::from_fn(|| {
        prefix.push("..");
        Some(prefix.join(wanted_path))
    })
    .take(max_parent_components.min(3));

    // Try without leading directories
    let mut trimmed_path = wanted_path;
    let remove = std::iter::from_fn(|| {
        let mut components = trimmed_path.components();
        let removed = components.next()?;
        trimmed_path = components.as_path();
        let _ = trimmed_path.file_name()?; // ensure there is a file name left
        Some([
            Some(trimmed_path.to_path_buf()),
            (removed != std::path::Component::ParentDir)
                .then(|| Path::new("..").join(trimmed_path)),
        ])
    })
    .flatten()
    .flatten()
    .take(4);

    root_absolute
        .chain(add)
        .chain(remove)
        .find(|new_path| source_map.file_exists(&base_dir.join(&new_path)))
}
