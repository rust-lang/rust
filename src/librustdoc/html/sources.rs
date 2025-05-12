use std::cell::RefCell;
use std::ffi::OsStr;
use std::path::{Component, Path, PathBuf};
use std::{fmt, fs};

use askama::Template;
use rustc_data_structures::fx::{FxHashSet, FxIndexMap};
use rustc_hir::def_id::LOCAL_CRATE;
use rustc_middle::ty::TyCtxt;
use rustc_session::Session;
use rustc_span::{FileName, FileNameDisplayPreference, RealFileName, sym};
use tracing::info;

use super::highlight;
use super::layout::{self, BufDisplay};
use super::render::Context;
use crate::clean;
use crate::clean::utils::has_doc_flag;
use crate::docfs::PathError;
use crate::error::Error;
use crate::visit::DocVisitor;

pub(crate) fn render(cx: &mut Context<'_>, krate: &clean::Crate) -> Result<(), Error> {
    info!("emitting source files");

    let dst = cx.dst.join("src").join(krate.name(cx.tcx()).as_str());
    cx.shared.ensure_dir(&dst)?;
    let crate_name = krate.name(cx.tcx());
    let crate_name = crate_name.as_str();

    let mut collector =
        SourceCollector { dst, cx, emitted_local_sources: FxHashSet::default(), crate_name };
    collector.visit_crate(krate);
    Ok(())
}

pub(crate) fn collect_local_sources(
    tcx: TyCtxt<'_>,
    src_root: &Path,
    krate: &clean::Crate,
) -> FxIndexMap<PathBuf, String> {
    let mut lsc = LocalSourcesCollector { tcx, local_sources: FxIndexMap::default(), src_root };
    lsc.visit_crate(krate);
    lsc.local_sources
}

struct LocalSourcesCollector<'a, 'tcx> {
    tcx: TyCtxt<'tcx>,
    local_sources: FxIndexMap<PathBuf, String>,
    src_root: &'a Path,
}

fn filename_real_and_local(span: clean::Span, sess: &Session) -> Option<RealFileName> {
    if span.cnum(sess) == LOCAL_CRATE
        && let FileName::Real(file) = span.filename(sess)
    {
        Some(file)
    } else {
        None
    }
}

impl LocalSourcesCollector<'_, '_> {
    fn add_local_source(&mut self, item: &clean::Item) {
        let sess = self.tcx.sess;
        let span = item.span(self.tcx);
        let Some(span) = span else { return };
        // skip all synthetic "files"
        let Some(p) = filename_real_and_local(span, sess).and_then(|file| file.into_local_path())
        else {
            return;
        };
        if self.local_sources.contains_key(&*p) {
            // We've already emitted this source
            return;
        }

        let href = RefCell::new(PathBuf::new());
        clean_path(
            self.src_root,
            &p,
            |component| {
                href.borrow_mut().push(component);
            },
            || {
                href.borrow_mut().pop();
            },
        );

        let mut href = href.into_inner().to_string_lossy().into_owned();
        if let Some(c) = href.as_bytes().last()
            && *c != b'/'
        {
            href.push('/');
        }
        let mut src_fname = p.file_name().expect("source has no filename").to_os_string();
        src_fname.push(".html");
        href.push_str(&src_fname.to_string_lossy());
        self.local_sources.insert(p, href);
    }
}

impl DocVisitor<'_> for LocalSourcesCollector<'_, '_> {
    fn visit_item(&mut self, item: &clean::Item) {
        self.add_local_source(item);

        self.visit_item_recur(item)
    }
}

/// Helper struct to render all source code to HTML pages
struct SourceCollector<'a, 'tcx> {
    cx: &'a mut Context<'tcx>,

    /// Root destination to place all HTML output into
    dst: PathBuf,
    emitted_local_sources: FxHashSet<PathBuf>,

    crate_name: &'a str,
}

impl DocVisitor<'_> for SourceCollector<'_, '_> {
    fn visit_item(&mut self, item: &clean::Item) {
        if !self.cx.info.include_sources {
            return;
        }

        let tcx = self.cx.tcx();
        let span = item.span(tcx);
        let Some(span) = span else { return };
        let sess = tcx.sess;

        // If we're not rendering sources, there's nothing to do.
        // If we're including source files, and we haven't seen this file yet,
        // then we need to render it out to the filesystem.
        if let Some(filename) = filename_real_and_local(span, sess) {
            let span = span.inner();
            let pos = sess.source_map().lookup_source_file(span.lo());
            let file_span = span.with_lo(pos.start_pos).with_hi(pos.end_position());
            // If it turns out that we couldn't read this file, then we probably
            // can't read any of the files (generating html output from json or
            // something like that), so just don't include sources for the
            // entire crate. The other option is maintaining this mapping on a
            // per-file basis, but that's probably not worth it...
            self.cx.info.include_sources = match self.emit_source(&filename, file_span) {
                Ok(()) => true,
                Err(e) => {
                    self.cx.shared.tcx.dcx().span_err(
                        span,
                        format!(
                            "failed to render source code for `{filename}`: {e}",
                            filename = filename.to_string_lossy(FileNameDisplayPreference::Local),
                        ),
                    );
                    false
                }
            };
        }

        self.visit_item_recur(item)
    }
}

impl SourceCollector<'_, '_> {
    /// Renders the given filename into its corresponding HTML source file.
    fn emit_source(
        &mut self,
        file: &RealFileName,
        file_span: rustc_span::Span,
    ) -> Result<(), Error> {
        let p = if let Some(local_path) = file.local_path() {
            local_path.to_path_buf()
        } else {
            unreachable!("only the current crate should have sources emitted");
        };
        if self.emitted_local_sources.contains(&*p) {
            // We've already emitted this source
            return Ok(());
        }

        let contents = match fs::read_to_string(&p) {
            Ok(contents) => contents,
            Err(e) => {
                return Err(Error::new(e, &p));
            }
        };

        // Remove the utf-8 BOM if any
        let contents = contents.strip_prefix('\u{feff}').unwrap_or(&contents);

        let shared = &self.cx.shared;
        // Create the intermediate directories
        let cur = RefCell::new(PathBuf::new());
        let root_path = RefCell::new(PathBuf::new());

        clean_path(
            &shared.src_root,
            &p,
            |component| {
                cur.borrow_mut().push(component);
                root_path.borrow_mut().push("..");
            },
            || {
                cur.borrow_mut().pop();
                root_path.borrow_mut().pop();
            },
        );

        let src_fname = p.file_name().expect("source has no filename").to_os_string();
        let mut fname = src_fname.clone();

        let root_path = PathBuf::from("../../").join(root_path.into_inner());
        let mut root_path = root_path.to_string_lossy();
        if let Some(c) = root_path.as_bytes().last()
            && *c != b'/'
        {
            root_path += "/";
        }
        let mut file_path = Path::new(&self.crate_name).join(&*cur.borrow());
        file_path.push(&fname);
        fname.push(".html");
        let mut cur = self.dst.join(cur.into_inner());
        shared.ensure_dir(&cur)?;

        cur.push(&fname);

        let title = format!("{} - source", src_fname.to_string_lossy());
        let desc = format!(
            "Source of the Rust file `{}`.",
            file.to_string_lossy(FileNameDisplayPreference::Remapped)
        );
        let page = layout::Page {
            title: &title,
            css_class: "src",
            root_path: &root_path,
            static_root_path: shared.static_root_path.as_deref(),
            description: &desc,
            resource_suffix: &shared.resource_suffix,
            rust_logo: has_doc_flag(self.cx.tcx(), LOCAL_CRATE.as_def_id(), sym::rust_logo),
        };
        let source_context = SourceContext::Standalone { file_path };
        let v = layout::render(
            &shared.layout,
            &page,
            "",
            BufDisplay(|buf: &mut String| {
                print_src(
                    buf,
                    contents,
                    file_span,
                    self.cx,
                    &root_path,
                    &highlight::DecorationInfo::default(),
                    &source_context,
                );
            }),
            &shared.style_files,
        );
        shared.fs.write(cur, v)?;
        self.emitted_local_sources.insert(p);
        Ok(())
    }
}

/// Takes a path to a source file and cleans the path to it. This canonicalizes
/// things like ".." to components which preserve the "top down" hierarchy of a
/// static HTML tree. Each component in the cleaned path will be passed as an
/// argument to `f`. The very last component of the path (ie the file name) is ignored.
/// If a `..` is encountered, the `parent` closure will be called to allow the callee to
/// handle it.
pub(crate) fn clean_path<F, P>(src_root: &Path, p: &Path, mut f: F, mut parent: P)
where
    F: FnMut(&OsStr),
    P: FnMut(),
{
    // make it relative, if possible
    let p = p.strip_prefix(src_root).unwrap_or(p);

    let mut iter = p.components().peekable();

    while let Some(c) = iter.next() {
        if iter.peek().is_none() {
            break;
        }

        match c {
            Component::ParentDir => parent(),
            Component::Normal(c) => f(c),
            _ => continue,
        }
    }
}

pub(crate) struct ScrapedInfo<'a> {
    pub(crate) offset: usize,
    pub(crate) name: &'a str,
    pub(crate) url: &'a str,
    pub(crate) title: &'a str,
    pub(crate) locations: String,
    pub(crate) needs_expansion: bool,
}

#[derive(Template)]
#[template(path = "scraped_source.html")]
struct ScrapedSource<'a, Code: std::fmt::Display> {
    info: &'a ScrapedInfo<'a>,
    code_html: Code,
    max_nb_digits: u32,
}

#[derive(Template)]
#[template(path = "source.html")]
struct Source<Code: std::fmt::Display> {
    code_html: Code,
    file_path: Option<(String, String)>,
    max_nb_digits: u32,
}

pub(crate) enum SourceContext<'a> {
    Standalone { file_path: PathBuf },
    Embedded(ScrapedInfo<'a>),
}

/// Wrapper struct to render the source code of a file. This will do things like
/// adding line numbers to the left-hand side.
pub(crate) fn print_src(
    mut writer: impl fmt::Write,
    s: &str,
    file_span: rustc_span::Span,
    context: &Context<'_>,
    root_path: &str,
    decoration_info: &highlight::DecorationInfo,
    source_context: &SourceContext<'_>,
) {
    let mut lines = s.lines().count();
    let line_info = if let SourceContext::Embedded(info) = source_context {
        highlight::LineInfo::new_scraped(lines as u32, info.offset as u32)
    } else {
        highlight::LineInfo::new(lines as u32)
    };
    if line_info.is_scraped_example {
        lines += line_info.start_line as usize;
    }
    let code = fmt::from_fn(move |fmt| {
        let current_href = context
            .href_from_span(clean::Span::new(file_span), false)
            .expect("only local crates should have sources emitted");
        highlight::write_code(
            fmt,
            s,
            Some(highlight::HrefContext { context, file_span, root_path, current_href }),
            Some(decoration_info),
            Some(line_info),
        );
        Ok(())
    });
    let max_nb_digits = if lines > 0 { lines.ilog(10) + 1 } else { 1 };
    match source_context {
        SourceContext::Standalone { file_path } => Source {
            code_html: code,
            file_path: if let Some(file_name) = file_path.file_name()
                && let Some(file_path) = file_path.parent()
            {
                Some((file_path.display().to_string(), file_name.display().to_string()))
            } else {
                None
            },
            max_nb_digits,
        }
        .render_into(&mut writer)
        .unwrap(),
        SourceContext::Embedded(info) => {
            ScrapedSource { info, code_html: code, max_nb_digits }
                .render_into(&mut writer)
                .unwrap();
        }
    };
}
