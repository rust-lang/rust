//! Script to check the validity of `href` links in our HTML documentation.
//!
//! In the past we've been quite error prone to writing in broken links as most
//! of them are manually rather than automatically added. As files move over
//! time or apis change old links become stale or broken. The purpose of this
//! script is to check all relative links in our documentation to make sure they
//! actually point to a valid place.
//!
//! Currently uses a combination of HTML parsing to
//! extract the `href` and `id` attributes,
//! and regex search on the original markdown to handle intra-doc links.
//!
//! These values are then translated to file URLs if possible and then the
//! destination is asserted to exist.
//!
//! A few exceptions are allowed as there's known bugs in rustdoc, but this
//! should catch the majority of "broken link" cases.

use std::cell::{Cell, RefCell};
use std::collections::hash_map::Entry;
use std::collections::{HashMap, HashSet};
use std::fs;
use std::iter::once;
use std::path::{Component, Path, PathBuf};
use std::rc::Rc;
use std::time::Instant;

use html5ever::tendril::ByteTendril;
use html5ever::tokenizer::{
    BufferQueue, TagToken, Token, TokenSink, TokenSinkResult, Tokenizer, TokenizerOpts,
};

// Add linkcheck exceptions here
// If at all possible you should use intra-doc links to avoid linkcheck issues. These
// are cases where that does not work
// [(generated_documentation_page, &[broken_links])]
#[rustfmt::skip]
const LINKCHECK_EXCEPTIONS: &[(&str, &[&str])] = &[
    // These try to link to std::collections, but are defined in alloc
    // https://github.com/rust-lang/rust/issues/74481
    ("std/collections/btree_map/struct.BTreeMap.html", &["#insert-and-complex-keys"]),
    ("std/collections/btree_set/struct.BTreeSet.html", &["#insert-and-complex-keys"]),
    ("alloc/collections/btree_map/struct.BTreeMap.html", &["#insert-and-complex-keys"]),
    ("alloc/collections/btree_set/struct.BTreeSet.html", &["#insert-and-complex-keys"]),

    // These try to link to various things in std, but are defined in core.
    // The docs in std::primitive use proper intra-doc links, so these seem fine to special-case.
    // Most these are broken because liballoc uses `#[lang_item]` magic to define things on
    // primitives that aren't available in core.
    ("alloc/slice/trait.Join.html", &["#method.join"]),
    ("alloc/slice/trait.Concat.html", &["#method.concat"]),
    ("alloc/slice/index.html", &["#method.concat", "#method.join"]),
    ("alloc/vec/struct.Vec.html", &["#method.sort_by_key", "#method.sort_by_cached_key"]),
    ("alloc/bstr/struct.ByteStr.html", &[
        "#method.to_ascii_uppercase",
        "#method.to_ascii_lowercase",
        "core/slice::sort_by_key",
        "core\\slice::sort_by_key",
        "#method.sort_by_cached_key",
        "#method.sort_by_key"
    ]),
    ("alloc/bstr/struct.ByteString.html", &[
        "#method.to_ascii_uppercase",
        "#method.to_ascii_lowercase",
        "core/slice::sort_by_key",
        "core\\slice::sort_by_key",
        "#method.sort_by_cached_key",
        "#method.sort_by_key"
    ]),
    ("core/bstr/struct.ByteStr.html", &[
        "#method.to_ascii_uppercase",
        "#method.to_ascii_lowercase",
        "core/bstr/slice::sort_by_key",
        "core\\bstr\\slice::sort_by_key",
        "#method.sort_by_cached_key"
    ]),
    ("core/primitive.str.html", &["#method.to_ascii_uppercase", "#method.to_ascii_lowercase"]),
    ("core/primitive.slice.html", &["#method.to_ascii_uppercase", "#method.to_ascii_lowercase",
                                    "core/slice::sort_by_key", "core\\slice::sort_by_key",
                                    "#method.sort_by_cached_key"]),
];

#[rustfmt::skip]
const INTRA_DOC_LINK_EXCEPTIONS: &[(&str, &[&str])] = &[
    // This is being used in the sense of 'inclusive range', not a markdown link
    ("core/ops/struct.RangeInclusive.html", &["begin</code>, <code>end"]),
    ("std/ops/struct.RangeInclusive.html", &["begin</code>, <code>end"]),
    ("core/range/legacy/struct.RangeInclusive.html", &["begin</code>, <code>end"]),
    ("std/range/legacy/struct.RangeInclusive.html", &["begin</code>, <code>end"]),
    ("core/slice/trait.SliceIndex.html", &["begin</code>, <code>end"]),
    ("alloc/slice/trait.SliceIndex.html", &["begin</code>, <code>end"]),
    ("std/slice/trait.SliceIndex.html", &["begin</code>, <code>end"]),
    ("core/primitive.str.html", &["begin</code>, <code>end"]),
    ("std/primitive.str.html", &["begin</code>, <code>end"]),

];

macro_rules! static_regex {
    ($re:literal) => {{
        static RE: ::std::sync::OnceLock<::regex::Regex> = ::std::sync::OnceLock::new();
        RE.get_or_init(|| ::regex::Regex::new($re).unwrap())
    }};
}

macro_rules! t {
    ($e:expr) => {
        match $e {
            Ok(e) => e,
            Err(e) => panic!("{} failed with {:?}", stringify!($e), e),
        }
    };
}

struct Cli {
    docs: PathBuf,
    link_targets_dirs: Vec<PathBuf>,
}

fn main() {
    let cli = match parse_cli() {
        Ok(cli) => cli,
        Err(err) => {
            eprintln!("error: {err}");
            usage_and_exit(1);
        }
    };

    let mut checker = Checker {
        root: cli.docs.clone(),
        link_targets_dirs: cli.link_targets_dirs,
        cache: HashMap::new(),
    };
    let mut report = Report {
        errors: 0,
        start: Instant::now(),
        html_files: 0,
        html_redirects: 0,
        links_checked: 0,
        links_ignored_external: 0,
        links_ignored_exception: 0,
        intra_doc_exceptions: 0,
        has_broken_urls: false,
    };
    checker.walk(&cli.docs, &mut report);
    report.report();
    if report.errors != 0 {
        println!("found some broken links");
        std::process::exit(1);
    }
}

fn parse_cli() -> Result<Cli, String> {
    fn to_absolute_path(arg: &str) -> Result<PathBuf, String> {
        std::path::absolute(arg).map_err(|e| format!("could not convert to absolute {arg}: {e}"))
    }

    let mut verbatim = false;
    let mut docs = None;
    let mut link_targets_dirs = Vec::new();

    let mut args = std::env::args().skip(1);
    while let Some(arg) = args.next() {
        if !verbatim && arg == "--" {
            verbatim = true;
        } else if !verbatim && (arg == "-h" || arg == "--help") {
            usage_and_exit(0)
        } else if !verbatim && arg == "--link-targets-dir" {
            link_targets_dirs.push(to_absolute_path(
                &args.next().ok_or("missing value for --link-targets-dir")?,
            )?);
        } else if !verbatim && let Some(value) = arg.strip_prefix("--link-targets-dir=") {
            link_targets_dirs.push(to_absolute_path(value)?);
        } else if !verbatim && arg.starts_with('-') {
            return Err(format!("unknown flag: {arg}"));
        } else if docs.is_none() {
            docs = Some(arg);
        } else {
            return Err("too many positional arguments".into());
        }
    }

    Ok(Cli {
        docs: to_absolute_path(&docs.ok_or("missing first positional argument")?)?,
        link_targets_dirs,
    })
}

fn usage_and_exit(code: i32) -> ! {
    eprintln!("usage: linkchecker PATH [--link-targets-dir=PATH ...]");
    std::process::exit(code)
}

struct Checker {
    root: PathBuf,
    link_targets_dirs: Vec<PathBuf>,
    cache: Cache,
}

struct Report {
    errors: u32,
    // Used to provide help message to remind the user to register a page in `SUMMARY.md`.
    has_broken_urls: bool,
    start: Instant,
    html_files: u32,
    html_redirects: u32,
    links_checked: u32,
    links_ignored_external: u32,
    links_ignored_exception: u32,
    intra_doc_exceptions: u32,
}

/// A cache entry.
enum FileEntry {
    /// An HTML file.
    ///
    /// This includes the contents of the HTML file, and an optional set of
    /// HTML IDs. The IDs are used for checking fragments. They are computed
    /// as-needed. The source is discarded (replaced with an empty string)
    /// after the file has been checked, to conserve on memory.
    HtmlFile { source: Rc<String>, ids: RefCell<HashSet<String>> },
    /// This file is an HTML redirect to the given local path.
    Redirect { target: PathBuf },
    /// This is not an HTML file.
    OtherFile,
    /// This is a directory.
    Dir,
    /// The file doesn't exist.
    Missing,
}

/// A cache to speed up file access.
type Cache = HashMap<String, FileEntry>;

fn small_url_encode(s: &str) -> String {
    urlencoding::encode(s).to_string()
}

impl Checker {
    /// Primary entry point for walking the filesystem to find HTML files to check.
    fn walk(&mut self, dir: &Path, report: &mut Report) {
        for entry in t!(dir.read_dir()).map(|e| t!(e)) {
            let path = entry.path();
            // Goes through symlinks
            let metadata = t!(fs::metadata(&path));
            if metadata.is_dir() {
                self.walk(&path, report);
            } else {
                self.check(&path, report);
            }
        }
    }

    /// Checks a single file.
    fn check(&mut self, file: &Path, report: &mut Report) {
        let (pretty_path, entry) = self.load_file(file, report);
        let source = match entry {
            FileEntry::Missing => panic!("missing file {:?} while walking", file),
            FileEntry::Dir => unreachable!("never with `check` path"),
            FileEntry::OtherFile => return,
            FileEntry::Redirect { .. } => return,
            FileEntry::HtmlFile { source, ids } => {
                parse_ids(&mut ids.borrow_mut(), &pretty_path, source, report);
                source.clone()
            }
        };

        let (base, urls) = get_urls(&source);
        for (i, url) in urls {
            self.check_url(file, &pretty_path, report, &base, i, &url);
        }

        self.check_intra_doc_links(file, &pretty_path, &source, report);

        // we don't need the source anymore,
        // so drop to reduce memory-usage
        match self.cache.get_mut(&pretty_path).unwrap() {
            FileEntry::HtmlFile { source, .. } => *source = Rc::new(String::new()),
            _ => unreachable!("must be html file"),
        }
    }

    fn check_url(
        &mut self,
        file: &Path,
        pretty_path: &str,
        report: &mut Report,
        base: &Option<String>,
        i: u64,
        url: &str,
    ) {
        // Ignore external URLs
        if url.starts_with("http:")
            || url.starts_with("https:")
            || url.starts_with("javascript:")
            || url.starts_with("ftp:")
            || url.starts_with("irc:")
            || url.starts_with("data:")
            || url.starts_with("mailto:")
        {
            report.links_ignored_external += 1;
            return;
        }
        report.links_checked += 1;
        let (url, fragment) = match url.split_once('#') {
            None => (url, None),
            Some((url, fragment)) => (url, Some(fragment)),
        };
        // NB: the `splitn` always succeeds, even if the delimiter is not present.
        let url = url.splitn(2, '?').next().unwrap();

        // Once we've plucked out the URL, parse it using our base url and
        // then try to extract a file path.
        let mut path = file.to_path_buf();
        if base.is_some() || !url.is_empty() {
            let base = base.as_deref().unwrap_or("");
            path.pop();
            for part in Path::new(base).join(url).components() {
                match part {
                    Component::Prefix(_) | Component::RootDir => {
                        // Avoid absolute paths as they make the docs not
                        // relocatable by making assumptions on where the docs
                        // are hosted relative to the site root.
                        report.errors += 1;
                        println!(
                            "{}:{}: absolute path - {}",
                            pretty_path,
                            i,
                            Path::new(base).join(url).display()
                        );
                        return;
                    }
                    Component::CurDir => {}
                    Component::ParentDir => {
                        path.pop();
                    }
                    Component::Normal(s) => {
                        path.push(s);
                    }
                }
            }
        }

        let (target_pretty_path, target_entry) = self.load_file(&path, report);
        let (target_source, target_ids) = match target_entry {
            FileEntry::Missing => {
                if is_exception(file, &target_pretty_path) {
                    report.links_ignored_exception += 1;
                } else {
                    report.errors += 1;
                    report.has_broken_urls = true;
                    println!("{}:{}: broken link - `{}`", pretty_path, i, target_pretty_path);
                }
                return;
            }
            FileEntry::Dir => {
                // Links to directories show as directory listings when viewing
                // the docs offline so it's best to avoid them.
                report.errors += 1;
                println!(
                    "{}:{}: directory link to `{}` \
                         (directory links should use index.html instead)",
                    pretty_path, i, target_pretty_path
                );
                return;
            }
            FileEntry::OtherFile => return,
            FileEntry::Redirect { target } => {
                let t = target.clone();
                let (target, redir_entry) = self.load_file(&t, report);
                match redir_entry {
                    FileEntry::Missing => {
                        report.errors += 1;
                        println!(
                            "{}:{}: broken redirect from `{}` to `{}`",
                            pretty_path, i, target_pretty_path, target
                        );
                        return;
                    }
                    FileEntry::Redirect { target } => {
                        // Redirect to a redirect, this link checker
                        // currently doesn't support this, since it would
                        // require cycle checking, etc.
                        report.errors += 1;
                        println!(
                            "{}:{}: redirect from `{}` to `{}` \
                                 which is also a redirect (not supported)",
                            pretty_path,
                            i,
                            target_pretty_path,
                            target.display()
                        );
                        return;
                    }
                    FileEntry::Dir => {
                        report.errors += 1;
                        println!(
                            "{}:{}: redirect from `{}` to `{}` \
                                 which is a directory \
                                 (directory links should use index.html instead)",
                            pretty_path, i, target_pretty_path, target
                        );
                        return;
                    }
                    FileEntry::OtherFile => return,
                    FileEntry::HtmlFile { source, ids } => (source, ids),
                }
            }
            FileEntry::HtmlFile { source, ids } => (source, ids),
        };

        // Alright, if we've found an HTML file for the target link. If
        // this is a fragment link, also check that the `id` exists.
        if let Some(ref fragment) = fragment {
            // Fragments like `#1-6` are most likely line numbers to be
            // interpreted by javascript, so we're ignoring these
            if fragment.splitn(2, '-').all(|f| f.chars().all(|c| c.is_numeric())) {
                return;
            }

            parse_ids(&mut target_ids.borrow_mut(), &pretty_path, target_source, report);

            if target_ids.borrow().contains(*fragment) {
                return;
            }

            if is_exception(file, &format!("#{}", fragment)) {
                report.links_ignored_exception += 1;
            } else {
                report.errors += 1;
                print!("{}:{}: broken link fragment ", pretty_path, i);
                println!("`#{}` pointing to `{}`", fragment, target_pretty_path);
            };
        }
    }

    fn check_intra_doc_links(
        &mut self,
        file: &Path,
        pretty_path: &str,
        source: &str,
        report: &mut Report,
    ) {
        let relative = file.strip_prefix(&self.root).expect("should always be relative to root");
        // Don't check the reference. It has several legitimate things that
        // look like [<code>…</code>]. The reference has its own broken link
        // checker in its CI which handles this using pulldown_cmark.
        //
        // This checks both the end of the root (when checking just the
        // reference directory) or the beginning (when checking all docs).
        if self.root.ends_with("reference") || relative.starts_with("reference") {
            return;
        }
        // Search for intra-doc links that rustdoc didn't warn about
        // NOTE: only looks at one line at a time; in practice this should find most links
        for (i, line) in source.lines().enumerate() {
            for broken_link in static_regex!(r#"\[<code>(.*)</code>\]"#).captures_iter(line) {
                if is_intra_doc_exception(file, &broken_link[1]) {
                    report.intra_doc_exceptions += 1;
                } else {
                    report.errors += 1;
                    print!("{}:{}: broken intra-doc link - ", pretty_path, i + 1);
                    println!("{}", &broken_link[0]);
                }
            }
        }
    }

    /// Load a file from disk, or from the cache if available.
    fn load_file(&mut self, file: &Path, report: &mut Report) -> (String, &FileEntry) {
        let pretty_path =
            file.strip_prefix(&self.root).unwrap_or(file).to_str().unwrap().to_string();

        for base in once(&self.root).chain(self.link_targets_dirs.iter()) {
            let entry = self.cache.entry(pretty_path.clone());
            if let Entry::Occupied(e) = &entry
                && !matches!(e.get(), FileEntry::Missing)
            {
                break;
            }

            let file = base.join(&pretty_path);
            entry.insert_entry(match fs::metadata(&file) {
                Ok(metadata) if metadata.is_dir() => FileEntry::Dir,
                Ok(_) => {
                    if file.extension().and_then(|s| s.to_str()) != Some("html") {
                        FileEntry::OtherFile
                    } else {
                        report.html_files += 1;
                        load_html_file(&file, report)
                    }
                }
                Err(e) if is_not_found_error(&file, &e) => FileEntry::Missing,
                Err(e) => panic!("unexpected read error for {}: {}", file.display(), e),
            });
        }

        let entry = self.cache.get(&pretty_path).unwrap();
        (pretty_path, entry)
    }
}

impl Report {
    fn report(&self) {
        println!("checked links in: {:.1}s", self.start.elapsed().as_secs_f64());
        println!("number of HTML files scanned: {}", self.html_files);
        println!("number of HTML redirects found: {}", self.html_redirects);
        println!("number of links checked: {}", self.links_checked);
        println!("number of links ignored due to external: {}", self.links_ignored_external);
        println!("number of links ignored due to exceptions: {}", self.links_ignored_exception);
        println!("number of intra doc links ignored: {}", self.intra_doc_exceptions);
        println!("errors found: {}", self.errors);

        if self.has_broken_urls {
            eprintln!(
                "NOTE: if you are adding or renaming a markdown file in a mdBook, don't forget to \
                register the page in SUMMARY.md"
            );
        }
    }
}

fn load_html_file(file: &Path, report: &mut Report) -> FileEntry {
    let source = match fs::read_to_string(file) {
        Ok(s) => Rc::new(s),
        Err(err) => {
            // This usually should not fail since `metadata` was already
            // called successfully on this file.
            panic!("unexpected read error for {}: {}", file.display(), err);
        }
    };
    match maybe_redirect(&source) {
        Some(target) => {
            report.html_redirects += 1;
            let target = file.parent().unwrap().join(target);
            FileEntry::Redirect { target }
        }
        None => FileEntry::HtmlFile { source: source.clone(), ids: RefCell::new(HashSet::new()) },
    }
}

fn is_intra_doc_exception(file: &Path, link: &str) -> bool {
    if let Some(entry) = INTRA_DOC_LINK_EXCEPTIONS.iter().find(|&(f, _)| file.ends_with(f)) {
        entry.1.is_empty() || entry.1.contains(&link)
    } else {
        false
    }
}

fn is_exception(file: &Path, link: &str) -> bool {
    if let Some(entry) = LINKCHECK_EXCEPTIONS.iter().find(|&(f, _)| file.ends_with(f)) {
        entry.1.contains(&link)
    } else {
        // FIXME(#63351): Concat trait in alloc/slice reexported in primitive page
        //
        // NOTE: This cannot be added to `LINKCHECK_EXCEPTIONS` because the resolved path
        // calculated in `check` function is outside `build/<triple>/doc` dir.
        // So the `strip_prefix` method just returns the old absolute broken path.
        if file.ends_with("std/primitive.slice.html") && link.ends_with("primitive.slice.html") {
            return true;
        }
        false
    }
}

/// If the given HTML file contents is an HTML redirect, this returns the
/// destination path given in the redirect.
fn maybe_redirect(source: &str) -> Option<String> {
    const REDIRECT_RUSTDOC: (usize, &str) = (7, "<p>Redirecting to <a href=");
    const REDIRECT_MDBOOK: (usize, &str) = (8 - 7, "<p>Redirecting to... <a href=");

    let mut lines = source.lines();

    let mut find_redirect = |(line_rel, redirect_pattern): (usize, &str)| {
        let redirect_line = lines.nth(line_rel)?;

        redirect_line.find(redirect_pattern).map(|i| {
            let rest = &redirect_line[(i + redirect_pattern.len() + 1)..];
            let pos_quote = rest.find('"').unwrap();
            rest[..pos_quote].to_owned()
        })
    };

    find_redirect(REDIRECT_RUSTDOC).or_else(|| find_redirect(REDIRECT_MDBOOK))
}

fn parse_html<Sink: TokenSink>(source: &str, sink: Sink) -> Sink {
    let tendril: ByteTendril = source.as_bytes().into();
    let mut input = BufferQueue::default();
    input.push_back(tendril.try_reinterpret().unwrap());

    let tok = Tokenizer::new(sink, TokenizerOpts::default());
    let _ = tok.feed(&mut input);
    assert!(input.is_empty());
    tok.end();
    tok.sink
}

#[derive(Default)]
struct AttrCollector {
    attr_name: &'static [u8],
    base: Cell<Option<String>>,
    found_attrs: RefCell<Vec<(u64, String)>>,
    /// Tracks whether or not it is inside a <script> tag.
    ///
    /// A lot of our sources have JSON script tags which have HTML embedded
    /// within, but that cannot be parsed or processed correctly (since it is
    /// JSON, not HTML). I think the sink is supposed to return
    /// `TokenSinkResult::Script(…)` (and then maybe switch parser?), but I
    /// don't fully understand the best way to use that, and this seems good
    /// enough for now.
    in_script: Cell<bool>,
}

impl TokenSink for AttrCollector {
    type Handle = ();

    fn process_token(&self, token: Token, line_number: u64) -> TokenSinkResult<()> {
        match token {
            TagToken(tag) => {
                let tag_name = tag.name.as_bytes();
                if tag_name == b"base" {
                    if let Some(href) =
                        tag.attrs.iter().find(|attr| attr.name.local.as_bytes() == b"href")
                    {
                        self.base.set(Some(href.value.to_string()));
                    }
                    return TokenSinkResult::Continue;
                } else if tag_name == b"script" {
                    self.in_script.set(!self.in_script.get());
                }
                if self.in_script.get() {
                    return TokenSinkResult::Continue;
                }
                for attr in tag.attrs.iter() {
                    let name = attr.name.local.as_bytes();
                    if name == self.attr_name {
                        let url = attr.value.to_string();
                        self.found_attrs.borrow_mut().push((line_number, url));
                    }
                }
            }
            // Note: ParseError is pretty noisy. It seems html5ever does not
            // particularly like some kinds of HTML comments.
            _ => {}
        }
        TokenSinkResult::Continue
    }
}

/// Retrieves href="..." attributes from HTML elements.
fn get_urls(source: &str) -> (Option<String>, Vec<(u64, String)>) {
    let collector = AttrCollector { attr_name: b"href", ..AttrCollector::default() };
    let sink = parse_html(source, collector);
    (sink.base.into_inner(), sink.found_attrs.into_inner())
}

/// Retrieves id="..." attributes from HTML elements.
fn parse_ids(ids: &mut HashSet<String>, file: &str, source: &str, report: &mut Report) {
    if !ids.is_empty() {
        // ids have already been parsed
        return;
    }

    let collector = AttrCollector { attr_name: b"id", ..AttrCollector::default() };
    let sink = parse_html(source, collector);
    for (line_number, id) in sink.found_attrs.into_inner() {
        let encoded = small_url_encode(&id);
        if let Some(id) = ids.replace(id) {
            report.errors += 1;
            println!("{}:{}: id is not unique: `{}`", file, line_number, id);
        }
        // Just in case, we also add the encoded id.
        ids.insert(encoded);
    }
}

fn is_not_found_error(path: &Path, error: &std::io::Error) -> bool {
    // https://docs.microsoft.com/en-us/windows/win32/debug/system-error-codes--0-499-
    const WINDOWS_ERROR_INVALID_NAME: i32 = 123;

    error.kind() == std::io::ErrorKind::NotFound
        // If a broken intra-doc link contains `::`, on windows, it will cause `ERROR_INVALID_NAME`
        // rather than `NotFound`. Explicitly check for that so that the broken link can be allowed
        // in `LINKCHECK_EXCEPTIONS`.
        || (cfg!(windows)
            && error.raw_os_error() == Some(WINDOWS_ERROR_INVALID_NAME)
            && path.as_os_str().to_str().map_or(false, |s| s.contains("::")))
}
