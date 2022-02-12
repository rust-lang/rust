//! Script to check the validity of `href` links in our HTML documentation.
//!
//! In the past we've been quite error prone to writing in broken links as most
//! of them are manually rather than automatically added. As files move over
//! time or apis change old links become stale or broken. The purpose of this
//! script is to check all relative links in our documentation to make sure they
//! actually point to a valid place.
//!
//! Currently this doesn't actually do any HTML parsing or anything fancy like
//! that, it just has a simple "regex" to search for `href` and `id` tags.
//! These values are then translated to file URLs if possible and then the
//! destination is asserted to exist.
//!
//! A few exceptions are allowed as there's known bugs in rustdoc, but this
//! should catch the majority of "broken link" cases.

use std::cell::RefCell;
use std::collections::{HashMap, HashSet};
use std::env;
use std::fs;
use std::io::ErrorKind;
use std::path::{Component, Path, PathBuf};
use std::rc::Rc;
use std::time::Instant;

use once_cell::sync::Lazy;
use regex::Regex;

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
    ("core/primitive.str.html", &["#method.to_ascii_uppercase", "#method.to_ascii_lowercase"]),
    ("core/primitive.slice.html", &["#method.to_ascii_uppercase", "#method.to_ascii_lowercase",
                                    "core/slice::sort_by_key", "core\\slice::sort_by_key",
                                    "#method.sort_by_cached_key"]),
];

#[rustfmt::skip]
const INTRA_DOC_LINK_EXCEPTIONS: &[(&str, &[&str])] = &[
    // This will never have links that are not in other pages.
    // To avoid repeating the exceptions twice, an empty list means all broken links are allowed.
    ("reference/print.html", &[]),
    // All the reference 'links' are actually ENBF highlighted as code
    ("reference/comments.html", &[
         "/</code> <code>!",
         "*</code> <code>!",
    ]),
    ("reference/identifiers.html", &[
         "a</code>-<code>z</code> <code>A</code>-<code>Z",
         "a</code>-<code>z</code> <code>A</code>-<code>Z</code> <code>0</code>-<code>9</code> <code>_",
         "a</code>-<code>z</code> <code>A</code>-<code>Z</code>] [<code>a</code>-<code>z</code> <code>A</code>-<code>Z</code> <code>0</code>-<code>9</code> <code>_",
    ]),
    ("reference/tokens.html", &[
         "0</code>-<code>1",
         "0</code>-<code>7",
         "0</code>-<code>9",
         "0</code>-<code>9",
         "0</code>-<code>9</code> <code>a</code>-<code>f</code> <code>A</code>-<code>F",
    ]),
    ("reference/notation.html", &[
         "b</code> <code>B",
         "a</code>-<code>z",
    ]),
    // This is being used in the sense of 'inclusive range', not a markdown link
    ("core/ops/struct.RangeInclusive.html", &["begin</code>, <code>end"]),
    ("std/ops/struct.RangeInclusive.html", &["begin</code>, <code>end"]),
    ("core/slice/trait.SliceIndex.html", &["begin</code>, <code>end"]),
    ("alloc/slice/trait.SliceIndex.html", &["begin</code>, <code>end"]),
    ("std/slice/trait.SliceIndex.html", &["begin</code>, <code>end"]),
    ("core/primitive.str.html", &["begin</code>, <code>end"]),
    ("std/primitive.str.html", &["begin</code>, <code>end"]),

];

static BROKEN_INTRA_DOC_LINK: Lazy<Regex> =
    Lazy::new(|| Regex::new(r#"\[<code>(.*)</code>\]"#).unwrap());

macro_rules! t {
    ($e:expr) => {
        match $e {
            Ok(e) => e,
            Err(e) => panic!("{} failed with {:?}", stringify!($e), e),
        }
    };
}

fn main() {
    let docs = env::args_os().nth(1).expect("doc path should be first argument");
    let docs = env::current_dir().unwrap().join(docs);
    let mut checker = Checker { root: docs.clone(), cache: HashMap::new() };
    let mut report = Report {
        errors: 0,
        start: Instant::now(),
        html_files: 0,
        html_redirects: 0,
        links_checked: 0,
        links_ignored_external: 0,
        links_ignored_exception: 0,
        intra_doc_exceptions: 0,
    };
    checker.walk(&docs, &mut report);
    report.report();
    if report.errors != 0 {
        println!("found some broken links");
        std::process::exit(1);
    }
}

struct Checker {
    root: PathBuf,
    cache: Cache,
}

struct Report {
    errors: u32,
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
    s.replace("<", "%3C")
        .replace(">", "%3E")
        .replace(" ", "%20")
        .replace("?", "%3F")
        .replace("'", "%27")
        .replace("&", "%26")
        .replace(",", "%2C")
        .replace(":", "%3A")
        .replace(";", "%3B")
        .replace("[", "%5B")
        .replace("]", "%5D")
        .replace("\"", "%22")
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

        // Search for anything that's the regex 'href[ ]*=[ ]*".*?"'
        with_attrs_in_source(&source, " href", |url, i, base| {
            // Ignore external URLs
            if url.starts_with("http:")
                || url.starts_with("https:")
                || url.starts_with("javascript:")
                || url.starts_with("ftp:")
                || url.starts_with("irc:")
                || url.starts_with("data:")
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
            if !base.is_empty() || !url.is_empty() {
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
                                i + 1,
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
                        println!(
                            "{}:{}: broken link - `{}`",
                            pretty_path,
                            i + 1,
                            target_pretty_path
                        );
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
                        pretty_path,
                        i + 1,
                        target_pretty_path
                    );
                    return;
                }
                FileEntry::OtherFile => return,
                FileEntry::Redirect { target } => {
                    let t = target.clone();
                    drop(target);
                    let (target, redir_entry) = self.load_file(&t, report);
                    match redir_entry {
                        FileEntry::Missing => {
                            report.errors += 1;
                            println!(
                                "{}:{}: broken redirect from `{}` to `{}`",
                                pretty_path,
                                i + 1,
                                target_pretty_path,
                                target
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
                                i + 1,
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
                                pretty_path,
                                i + 1,
                                target_pretty_path,
                                target
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

                // These appear to be broken in mdbook right now?
                if fragment.starts_with('-') {
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
                    print!("{}:{}: broken link fragment ", pretty_path, i + 1);
                    println!("`#{}` pointing to `{}`", fragment, target_pretty_path);
                };
            }
        });

        // Search for intra-doc links that rustdoc didn't warn about
        // FIXME(#77199, 77200) Rustdoc should just warn about these directly.
        // NOTE: only looks at one line at a time; in practice this should find most links
        for (i, line) in source.lines().enumerate() {
            for broken_link in BROKEN_INTRA_DOC_LINK.captures_iter(line) {
                if is_intra_doc_exception(file, &broken_link[1]) {
                    report.intra_doc_exceptions += 1;
                } else {
                    report.errors += 1;
                    print!("{}:{}: broken intra-doc link - ", pretty_path, i + 1);
                    println!("{}", &broken_link[0]);
                }
            }
        }
        // we don't need the source anymore,
        // so drop to reduce memory-usage
        match self.cache.get_mut(&pretty_path).unwrap() {
            FileEntry::HtmlFile { source, .. } => *source = Rc::new(String::new()),
            _ => unreachable!("must be html file"),
        }
    }

    /// Load a file from disk, or from the cache if available.
    fn load_file(&mut self, file: &Path, report: &mut Report) -> (String, &FileEntry) {
        // https://docs.microsoft.com/en-us/windows/win32/debug/system-error-codes--0-499-
        #[cfg(windows)]
        const ERROR_INVALID_NAME: i32 = 123;

        let pretty_path =
            file.strip_prefix(&self.root).unwrap_or(&file).to_str().unwrap().to_string();

        let entry =
            self.cache.entry(pretty_path.clone()).or_insert_with(|| match fs::metadata(file) {
                Ok(metadata) if metadata.is_dir() => FileEntry::Dir,
                Ok(_) => {
                    if file.extension().and_then(|s| s.to_str()) != Some("html") {
                        FileEntry::OtherFile
                    } else {
                        report.html_files += 1;
                        load_html_file(file, report)
                    }
                }
                Err(e) if e.kind() == ErrorKind::NotFound => FileEntry::Missing,
                Err(e) => {
                    // If a broken intra-doc link contains `::`, on windows, it will cause `ERROR_INVALID_NAME` rather than `NotFound`.
                    // Explicitly check for that so that the broken link can be allowed in `LINKCHECK_EXCEPTIONS`.
                    #[cfg(windows)]
                    if e.raw_os_error() == Some(ERROR_INVALID_NAME)
                        && file.as_os_str().to_str().map_or(false, |s| s.contains("::"))
                    {
                        return FileEntry::Missing;
                    }
                    panic!("unexpected read error for {}: {}", file.display(), e);
                }
            });
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
        if file.ends_with("std/primitive.slice.html") {
            if link.ends_with("primitive.slice.html") {
                return true;
            }
        }
        false
    }
}

/// If the given HTML file contents is an HTML redirect, this returns the
/// destination path given in the redirect.
fn maybe_redirect(source: &str) -> Option<String> {
    const REDIRECT: &str = "<p>Redirecting to <a href=";

    let mut lines = source.lines();
    let redirect_line = lines.nth(7)?;

    redirect_line.find(REDIRECT).map(|i| {
        let rest = &redirect_line[(i + REDIRECT.len() + 1)..];
        let pos_quote = rest.find('"').unwrap();
        rest[..pos_quote].to_owned()
    })
}

fn with_attrs_in_source<F: FnMut(&str, usize, &str)>(source: &str, attr: &str, mut f: F) {
    let mut base = "";
    for (i, mut line) in source.lines().enumerate() {
        while let Some(j) = line.find(attr) {
            let rest = &line[j + attr.len()..];
            // The base tag should always be the first link in the document so
            // we can get away with using one pass.
            let is_base = line[..j].ends_with("<base");
            line = rest;
            let pos_equals = match rest.find('=') {
                Some(i) => i,
                None => continue,
            };
            if rest[..pos_equals].trim_start_matches(' ') != "" {
                continue;
            }

            let rest = &rest[pos_equals + 1..];

            let pos_quote = match rest.find(&['"', '\''][..]) {
                Some(i) => i,
                None => continue,
            };
            let quote_delim = rest.as_bytes()[pos_quote] as char;

            if rest[..pos_quote].trim_start_matches(' ') != "" {
                continue;
            }
            let rest = &rest[pos_quote + 1..];
            let url = match rest.find(quote_delim) {
                Some(i) => &rest[..i],
                None => continue,
            };
            if is_base {
                base = url;
                continue;
            }
            f(url, i, base)
        }
    }
}

fn parse_ids(ids: &mut HashSet<String>, file: &str, source: &str, report: &mut Report) {
    if ids.is_empty() {
        with_attrs_in_source(source, " id", |fragment, i, _| {
            let frag = fragment.trim_start_matches("#").to_owned();
            let encoded = small_url_encode(&frag);
            if !ids.insert(frag) {
                report.errors += 1;
                println!("{}:{}: id is not unique: `{}`", file, i, fragment);
            }
            // Just in case, we also add the encoded id.
            ids.insert(encoded);
        });
    }
}
