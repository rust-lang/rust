use std::path::Path;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

use rustc_data_structures::sync::IntoDynSyncSend;
use rustc_errors::emitter::{DynEmitter, Emitter, HumanEmitter, SilentEmitter, stderr_destination};
use rustc_errors::registry::Registry;
use rustc_errors::translation::Translator;
use rustc_errors::{ColorConfig, Diag, DiagCtxt, DiagInner, Level as DiagnosticLevel};
use rustc_session::parse::ParseSess as RawParseSess;
use rustc_span::{
    BytePos, Span,
    source_map::{FilePathMapping, SourceMap},
    symbol,
};

use crate::config::file_lines::LineRange;
use crate::config::options::Color;
use crate::ignore_path::IgnorePathSet;
use crate::parse::parser::{ModError, ModulePathSuccess};
use crate::source_map::LineRangeUtils;
use crate::utils::starts_with_newline;
use crate::visitor::SnippetProvider;
use crate::{Config, ErrorKind, FileName};

/// ParseSess holds structs necessary for constructing a parser.
pub(crate) struct ParseSess {
    raw_psess: RawParseSess,
    ignore_path_set: Arc<IgnorePathSet>,
    can_reset_errors: Arc<AtomicBool>,
}

/// Emit errors against every files expect ones specified in the `ignore_path_set`.
struct SilentOnIgnoredFilesEmitter {
    ignore_path_set: IntoDynSyncSend<Arc<IgnorePathSet>>,
    source_map: Arc<SourceMap>,
    emitter: Box<DynEmitter>,
    has_non_ignorable_parser_errors: bool,
    can_reset: Arc<AtomicBool>,
}

impl SilentOnIgnoredFilesEmitter {
    fn handle_non_ignoreable_error(&mut self, diag: DiagInner, registry: &Registry) {
        self.has_non_ignorable_parser_errors = true;
        self.can_reset.store(false, Ordering::Release);
        self.emitter.emit_diagnostic(diag, registry);
    }
}

impl Emitter for SilentOnIgnoredFilesEmitter {
    fn source_map(&self) -> Option<&SourceMap> {
        None
    }

    fn emit_diagnostic(&mut self, diag: DiagInner, registry: &Registry) {
        if diag.level() == DiagnosticLevel::Fatal {
            return self.handle_non_ignoreable_error(diag, registry);
        }
        if let Some(primary_span) = &diag.span.primary_span() {
            let file_name = self.source_map.span_to_filename(*primary_span);
            if let rustc_span::FileName::Real(rustc_span::RealFileName::LocalPath(ref path)) =
                file_name
            {
                if self
                    .ignore_path_set
                    .is_match(&FileName::Real(path.to_path_buf()))
                {
                    if !self.has_non_ignorable_parser_errors {
                        self.can_reset.store(true, Ordering::Release);
                    }
                    return;
                }
            };
        }
        self.handle_non_ignoreable_error(diag, registry);
    }

    fn translator(&self) -> &Translator {
        self.emitter.translator()
    }
}

impl From<Color> for ColorConfig {
    fn from(color: Color) -> Self {
        match color {
            Color::Auto => ColorConfig::Auto,
            Color::Always => ColorConfig::Always,
            Color::Never => ColorConfig::Never,
        }
    }
}

fn default_dcx(
    source_map: Arc<SourceMap>,
    ignore_path_set: Arc<IgnorePathSet>,
    can_reset: Arc<AtomicBool>,
    show_parse_errors: bool,
    color: Color,
) -> DiagCtxt {
    let supports_color = term::stderr().map_or(false, |term| term.supports_color());
    let emit_color = if supports_color {
        ColorConfig::from(color)
    } else {
        ColorConfig::Never
    };

    let translator = rustc_driver::default_translator();

    let emitter: Box<DynEmitter> = if show_parse_errors {
        Box::new(
            HumanEmitter::new(stderr_destination(emit_color), translator)
                .sm(Some(source_map.clone())),
        )
    } else {
        Box::new(SilentEmitter { translator })
    };
    DiagCtxt::new(Box::new(SilentOnIgnoredFilesEmitter {
        has_non_ignorable_parser_errors: false,
        source_map,
        emitter,
        ignore_path_set: IntoDynSyncSend(ignore_path_set),
        can_reset,
    }))
}

impl ParseSess {
    pub(crate) fn new(config: &Config) -> Result<ParseSess, ErrorKind> {
        let ignore_path_set = match IgnorePathSet::from_ignore_list(&config.ignore()) {
            Ok(ignore_path_set) => Arc::new(ignore_path_set),
            Err(e) => return Err(ErrorKind::InvalidGlobPattern(e)),
        };
        let source_map = Arc::new(SourceMap::new(FilePathMapping::empty()));
        let can_reset_errors = Arc::new(AtomicBool::new(false));

        let dcx = default_dcx(
            Arc::clone(&source_map),
            Arc::clone(&ignore_path_set),
            Arc::clone(&can_reset_errors),
            config.show_parse_errors(),
            config.color(),
        );
        let raw_psess = RawParseSess::with_dcx(dcx, source_map);

        Ok(ParseSess {
            raw_psess,
            ignore_path_set,
            can_reset_errors,
        })
    }

    /// Determine the submodule path for the given module identifier.
    ///
    /// * `id` - The name of the module
    /// * `relative` - If Some(symbol), the symbol name is a directory relative to the dir_path.
    ///   If relative is Some, resolve the submodule at {dir_path}/{symbol}/{id}.rs
    ///   or {dir_path}/{symbol}/{id}/mod.rs. if None, resolve the module at {dir_path}/{id}.rs.
    /// *  `dir_path` - Module resolution will occur relative to this directory.
    pub(crate) fn default_submod_path(
        &self,
        id: symbol::Ident,
        relative: Option<symbol::Ident>,
        dir_path: &Path,
    ) -> Result<ModulePathSuccess, ModError<'_>> {
        rustc_expand::module::default_submod_path(&self.raw_psess, id, relative, dir_path).or_else(
            |e| {
                // If resolving a module relative to {dir_path}/{symbol} fails because a file
                // could not be found, then try to resolve the module relative to {dir_path}.
                // If we still can't find the module after searching for it in {dir_path},
                // surface the original error.
                if matches!(e, ModError::FileNotFound(..)) && relative.is_some() {
                    rustc_expand::module::default_submod_path(&self.raw_psess, id, None, dir_path)
                        .map_err(|_| e)
                } else {
                    Err(e)
                }
            },
        )
    }

    pub(crate) fn is_file_parsed(&self, path: &Path) -> bool {
        self.raw_psess
            .source_map()
            .get_source_file(&rustc_span::FileName::Real(
                rustc_span::RealFileName::LocalPath(path.to_path_buf()),
            ))
            .is_some()
    }

    pub(crate) fn ignore_file(&self, path: &FileName) -> bool {
        self.ignore_path_set.as_ref().is_match(path)
    }

    pub(crate) fn set_silent_emitter(&mut self) {
        self.raw_psess.dcx().make_silent();
    }

    pub(crate) fn span_to_filename(&self, span: Span) -> FileName {
        self.raw_psess.source_map().span_to_filename(span).into()
    }

    pub(crate) fn span_to_file_contents(&self, span: Span) -> Arc<rustc_span::SourceFile> {
        self.raw_psess
            .source_map()
            .lookup_source_file(span.data().lo)
    }

    pub(crate) fn span_to_first_line_string(&self, span: Span) -> String {
        let file_lines = self.raw_psess.source_map().span_to_lines(span).ok();

        match file_lines {
            Some(fl) => fl
                .file
                .get_line(fl.lines[0].line_index)
                .map_or_else(String::new, |s| s.to_string()),
            None => String::new(),
        }
    }

    pub(crate) fn line_of_byte_pos(&self, pos: BytePos) -> usize {
        self.raw_psess.source_map().lookup_char_pos(pos).line
    }

    // TODO(calebcartwright): Preemptive, currently unused addition
    // that will be used to support formatting scenarios that take original
    // positions into account
    /// Determines whether two byte positions are in the same source line.
    #[allow(dead_code)]
    pub(crate) fn byte_pos_same_line(&self, a: BytePos, b: BytePos) -> bool {
        self.line_of_byte_pos(a) == self.line_of_byte_pos(b)
    }

    pub(crate) fn span_to_debug_info(&self, span: Span) -> String {
        self.raw_psess.source_map().span_to_diagnostic_string(span)
    }

    pub(crate) fn inner(&self) -> &RawParseSess {
        &self.raw_psess
    }

    pub(crate) fn snippet_provider(&self, span: Span) -> SnippetProvider {
        let source_file = self.raw_psess.source_map().lookup_char_pos(span.lo()).file;
        SnippetProvider::new(
            source_file.start_pos,
            source_file.end_position(),
            Arc::clone(source_file.src.as_ref().unwrap()),
        )
    }

    pub(crate) fn get_original_snippet(&self, file_name: &FileName) -> Option<Arc<String>> {
        self.raw_psess
            .source_map()
            .get_source_file(&file_name.into())
            .and_then(|source_file| source_file.src.clone())
    }
}

// Methods that should be restricted within the parse module.
impl ParseSess {
    pub(super) fn emit_diagnostics(&self, diagnostics: Vec<Diag<'_>>) {
        for diagnostic in diagnostics {
            diagnostic.emit();
        }
    }

    pub(super) fn can_reset_errors(&self) -> bool {
        self.can_reset_errors.load(Ordering::Acquire)
    }

    pub(super) fn has_errors(&self) -> bool {
        self.raw_psess.dcx().has_errors().is_some()
    }

    pub(super) fn reset_errors(&self) {
        self.raw_psess.dcx().reset_err_count();
    }
}

impl LineRangeUtils for ParseSess {
    fn lookup_line_range(&self, span: Span) -> LineRange {
        let snippet = self
            .raw_psess
            .source_map()
            .span_to_snippet(span)
            .unwrap_or_default();
        let lo = self.raw_psess.source_map().lookup_line(span.lo()).unwrap();
        let hi = self.raw_psess.source_map().lookup_line(span.hi()).unwrap();

        debug_assert_eq!(
            lo.sf.name, hi.sf.name,
            "span crossed file boundary: lo: {lo:?}, hi: {hi:?}"
        );

        // in case the span starts with a newline, the line range is off by 1 without the
        // adjustment below
        let offset = 1 + if starts_with_newline(&snippet) { 1 } else { 0 };
        // Line numbers start at 1
        LineRange {
            file: lo.sf.clone(),
            lo: lo.line + offset,
            hi: hi.line + offset,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use rustfmt_config_proc_macro::nightly_only_test;

    mod emitter {
        use super::*;
        use crate::config::IgnoreList;
        use crate::utils::mk_sp;
        use rustc_errors::MultiSpan;
        use rustc_span::{FileName as SourceMapFileName, RealFileName};
        use std::path::PathBuf;
        use std::sync::atomic::AtomicU32;

        struct TestEmitter {
            num_emitted_errors: Arc<AtomicU32>,
        }

        impl Emitter for TestEmitter {
            fn source_map(&self) -> Option<&SourceMap> {
                None
            }

            fn emit_diagnostic(&mut self, _diag: DiagInner, _registry: &Registry) {
                self.num_emitted_errors.fetch_add(1, Ordering::Release);
            }

            fn translator(&self) -> &Translator {
                panic!("test emitter attempted to translate a diagnostic");
            }
        }

        fn build_diagnostic(level: DiagnosticLevel, span: Option<MultiSpan>) -> DiagInner {
            #[allow(rustc::untranslatable_diagnostic)] // no translation needed for empty string
            let mut diag = DiagInner::new(level, "");
            diag.messages.clear();
            if let Some(span) = span {
                diag.span = span;
            }
            diag
        }

        fn build_emitter(
            num_emitted_errors: Arc<AtomicU32>,
            can_reset: Arc<AtomicBool>,
            source_map: Option<Arc<SourceMap>>,
            ignore_list: Option<IgnoreList>,
        ) -> SilentOnIgnoredFilesEmitter {
            let emitter_writer = TestEmitter { num_emitted_errors };
            let source_map =
                source_map.unwrap_or_else(|| Arc::new(SourceMap::new(FilePathMapping::empty())));
            let ignore_path_set = Arc::new(
                IgnorePathSet::from_ignore_list(&ignore_list.unwrap_or_default()).unwrap(),
            );
            SilentOnIgnoredFilesEmitter {
                has_non_ignorable_parser_errors: false,
                source_map,
                emitter: Box::new(emitter_writer),
                ignore_path_set: IntoDynSyncSend(ignore_path_set),
                can_reset,
            }
        }

        fn get_ignore_list(config: &str) -> IgnoreList {
            Config::from_toml(config, Path::new("./rustfmt.toml"))
                .unwrap()
                .ignore()
        }

        #[test]
        fn handles_fatal_parse_error_in_ignored_file() {
            let num_emitted_errors = Arc::new(AtomicU32::new(0));
            let can_reset_errors = Arc::new(AtomicBool::new(false));
            let ignore_list = get_ignore_list(r#"ignore = ["foo.rs"]"#);
            let source_map = Arc::new(SourceMap::new(FilePathMapping::empty()));
            let source =
                String::from(r#"extern "system" fn jni_symbol!( funcName ) ( ... ) -> {} "#);
            source_map.new_source_file(
                SourceMapFileName::Real(RealFileName::LocalPath(PathBuf::from("foo.rs"))),
                source,
            );
            let registry = Registry::new(&[]);
            let mut emitter = build_emitter(
                Arc::clone(&num_emitted_errors),
                Arc::clone(&can_reset_errors),
                Some(Arc::clone(&source_map)),
                Some(ignore_list),
            );
            let span = MultiSpan::from_span(mk_sp(BytePos(0), BytePos(1)));
            let fatal_diagnostic = build_diagnostic(DiagnosticLevel::Fatal, Some(span));
            emitter.emit_diagnostic(fatal_diagnostic, &registry);
            assert_eq!(num_emitted_errors.load(Ordering::Acquire), 1);
            assert_eq!(can_reset_errors.load(Ordering::Acquire), false);
        }

        #[nightly_only_test]
        #[test]
        fn handles_recoverable_parse_error_in_ignored_file() {
            let num_emitted_errors = Arc::new(AtomicU32::new(0));
            let can_reset_errors = Arc::new(AtomicBool::new(false));
            let ignore_list = get_ignore_list(r#"ignore = ["foo.rs"]"#);
            let source_map = Arc::new(SourceMap::new(FilePathMapping::empty()));
            let source = String::from(r#"pub fn bar() { 1x; }"#);
            source_map.new_source_file(
                SourceMapFileName::Real(RealFileName::LocalPath(PathBuf::from("foo.rs"))),
                source,
            );
            let registry = Registry::new(&[]);
            let mut emitter = build_emitter(
                Arc::clone(&num_emitted_errors),
                Arc::clone(&can_reset_errors),
                Some(Arc::clone(&source_map)),
                Some(ignore_list),
            );
            let span = MultiSpan::from_span(mk_sp(BytePos(0), BytePos(1)));
            let non_fatal_diagnostic = build_diagnostic(DiagnosticLevel::Warning, Some(span));
            emitter.emit_diagnostic(non_fatal_diagnostic, &registry);
            assert_eq!(num_emitted_errors.load(Ordering::Acquire), 0);
            assert_eq!(can_reset_errors.load(Ordering::Acquire), true);
        }

        #[nightly_only_test]
        #[test]
        fn handles_recoverable_parse_error_in_non_ignored_file() {
            let num_emitted_errors = Arc::new(AtomicU32::new(0));
            let can_reset_errors = Arc::new(AtomicBool::new(false));
            let source_map = Arc::new(SourceMap::new(FilePathMapping::empty()));
            let source = String::from(r#"pub fn bar() { 1x; }"#);
            source_map.new_source_file(
                SourceMapFileName::Real(RealFileName::LocalPath(PathBuf::from("foo.rs"))),
                source,
            );
            let registry = Registry::new(&[]);
            let mut emitter = build_emitter(
                Arc::clone(&num_emitted_errors),
                Arc::clone(&can_reset_errors),
                Some(Arc::clone(&source_map)),
                None,
            );
            let span = MultiSpan::from_span(mk_sp(BytePos(0), BytePos(1)));
            let non_fatal_diagnostic = build_diagnostic(DiagnosticLevel::Warning, Some(span));
            emitter.emit_diagnostic(non_fatal_diagnostic, &registry);
            assert_eq!(num_emitted_errors.load(Ordering::Acquire), 1);
            assert_eq!(can_reset_errors.load(Ordering::Acquire), false);
        }

        #[nightly_only_test]
        #[test]
        fn handles_mix_of_recoverable_parse_error() {
            let num_emitted_errors = Arc::new(AtomicU32::new(0));
            let can_reset_errors = Arc::new(AtomicBool::new(false));
            let source_map = Arc::new(SourceMap::new(FilePathMapping::empty()));
            let ignore_list = get_ignore_list(r#"ignore = ["foo.rs"]"#);
            let bar_source = String::from(r#"pub fn bar() { 1x; }"#);
            let foo_source = String::from(r#"pub fn foo() { 1x; }"#);
            let fatal_source =
                String::from(r#"extern "system" fn jni_symbol!( funcName ) ( ... ) -> {} "#);
            source_map.new_source_file(
                SourceMapFileName::Real(RealFileName::LocalPath(PathBuf::from("bar.rs"))),
                bar_source,
            );
            source_map.new_source_file(
                SourceMapFileName::Real(RealFileName::LocalPath(PathBuf::from("foo.rs"))),
                foo_source,
            );
            source_map.new_source_file(
                SourceMapFileName::Real(RealFileName::LocalPath(PathBuf::from("fatal.rs"))),
                fatal_source,
            );
            let registry = Registry::new(&[]);
            let mut emitter = build_emitter(
                Arc::clone(&num_emitted_errors),
                Arc::clone(&can_reset_errors),
                Some(Arc::clone(&source_map)),
                Some(ignore_list),
            );
            let bar_span = MultiSpan::from_span(mk_sp(BytePos(0), BytePos(1)));
            let foo_span = MultiSpan::from_span(mk_sp(BytePos(21), BytePos(22)));
            let bar_diagnostic = build_diagnostic(DiagnosticLevel::Warning, Some(bar_span));
            let foo_diagnostic = build_diagnostic(DiagnosticLevel::Warning, Some(foo_span));
            let fatal_diagnostic = build_diagnostic(DiagnosticLevel::Fatal, None);
            emitter.emit_diagnostic(bar_diagnostic, &registry);
            emitter.emit_diagnostic(foo_diagnostic, &registry);
            emitter.emit_diagnostic(fatal_diagnostic, &registry);
            assert_eq!(num_emitted_errors.load(Ordering::Acquire), 2);
            assert_eq!(can_reset_errors.load(Ordering::Acquire), false);
        }
    }
}
