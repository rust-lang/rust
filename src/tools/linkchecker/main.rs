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

use std::collections::hash_map::Entry;
use std::collections::{HashMap, HashSet};
use std::env;
use std::fs;
use std::path::{Component, Path, PathBuf};
use std::rc::Rc;

use once_cell::sync::Lazy;
use regex::Regex;

use crate::Redirect::*;

// Add linkcheck exceptions here
// If at all possible you should use intra-doc links to avoid linkcheck issues. These
// are cases where that does not work
// [(generated_documentation_page, &[broken_links])]
const LINKCHECK_EXCEPTIONS: &[(&str, &[&str])] = &[
    // These are methods on slice, and `Self` does not work on primitive impls
    // in intra-doc links (primitive impls are weird)
    // https://github.com/rust-lang/rust/issues/62834 is necessary to be
    // able to link to slices
    (
        "std/io/struct.IoSlice.html",
        &[
            "#method.as_mut_ptr",
            "#method.sort_by_key",
            "#method.make_ascii_uppercase",
            "#method.make_ascii_lowercase",
            "#method.get_unchecked_mut",
        ],
    ),
    // These try to link to std::collections, but are defined in alloc
    // https://github.com/rust-lang/rust/issues/74481
    ("std/collections/btree_map/struct.BTreeMap.html", &["#insert-and-complex-keys"]),
    ("std/collections/btree_set/struct.BTreeSet.html", &["#insert-and-complex-keys"]),
    ("alloc/collections/btree_map/struct.BTreeMap.html", &["#insert-and-complex-keys"]),
    ("alloc/collections/btree_set/struct.BTreeSet.html", &["#insert-and-complex-keys"]),
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
    let docs = env::args_os().nth(1).unwrap();
    let docs = env::current_dir().unwrap().join(docs);
    let mut errors = false;
    walk(&mut HashMap::new(), &docs, &docs, &mut errors);
    if errors {
        panic!("found some broken links");
    }
}

#[derive(Debug)]
pub enum LoadError {
    IOError(std::io::Error),
    BrokenRedirect(PathBuf, std::io::Error),
    IsRedirect,
}

enum Redirect {
    SkipRedirect,
    FromRedirect(bool),
}

struct FileEntry {
    source: Rc<String>,
    ids: HashSet<String>,
}

type Cache = HashMap<PathBuf, FileEntry>;

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

impl FileEntry {
    fn parse_ids(&mut self, file: &Path, contents: &str, errors: &mut bool) {
        if self.ids.is_empty() {
            with_attrs_in_source(contents, " id", |fragment, i, _| {
                let frag = fragment.trim_start_matches("#").to_owned();
                let encoded = small_url_encode(&frag);
                if !self.ids.insert(frag) {
                    *errors = true;
                    println!("{}:{}: id is not unique: `{}`", file.display(), i, fragment);
                }
                // Just in case, we also add the encoded id.
                self.ids.insert(encoded);
            });
        }
    }
}

fn walk(cache: &mut Cache, root: &Path, dir: &Path, errors: &mut bool) {
    for entry in t!(dir.read_dir()).map(|e| t!(e)) {
        let path = entry.path();
        let kind = t!(entry.file_type());
        if kind.is_dir() {
            walk(cache, root, &path, errors);
        } else {
            let pretty_path = check(cache, root, &path, errors);
            if let Some(pretty_path) = pretty_path {
                let entry = cache.get_mut(&pretty_path).unwrap();
                // we don't need the source anymore,
                // so drop to reduce memory-usage
                entry.source = Rc::new(String::new());
            }
        }
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

fn check(cache: &mut Cache, root: &Path, file: &Path, errors: &mut bool) -> Option<PathBuf> {
    // Ignore non-HTML files.
    if file.extension().and_then(|s| s.to_str()) != Some("html") {
        return None;
    }

    let res = load_file(cache, root, file, SkipRedirect);
    let (pretty_file, contents) = match res {
        Ok(res) => res,
        Err(_) => return None,
    };
    {
        cache.get_mut(&pretty_file).unwrap().parse_ids(&pretty_file, &contents, errors);
    }

    // Search for anything that's the regex 'href[ ]*=[ ]*".*?"'
    with_attrs_in_source(&contents, " href", |url, i, base| {
        // Ignore external URLs
        if url.starts_with("http:")
            || url.starts_with("https:")
            || url.starts_with("javascript:")
            || url.starts_with("ftp:")
            || url.starts_with("irc:")
            || url.starts_with("data:")
        {
            return;
        }
        let mut parts = url.splitn(2, '#');
        let url = parts.next().unwrap();
        let fragment = parts.next();
        let mut parts = url.splitn(2, '?');
        let url = parts.next().unwrap();

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
                        *errors = true;
                        println!(
                            "{}:{}: absolute path - {}",
                            pretty_file.display(),
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

        // Alright, if we've found a file name then this file had better
        // exist! If it doesn't then we register and print an error.
        if path.exists() {
            if path.is_dir() {
                // Links to directories show as directory listings when viewing
                // the docs offline so it's best to avoid them.
                *errors = true;
                let pretty_path = path.strip_prefix(root).unwrap_or(&path);
                println!(
                    "{}:{}: directory link - {}",
                    pretty_file.display(),
                    i + 1,
                    pretty_path.display()
                );
                return;
            }
            if let Some(extension) = path.extension() {
                // Ignore none HTML files.
                if extension != "html" {
                    return;
                }
            }
            let res = load_file(cache, root, &path, FromRedirect(false));
            let (pretty_path, contents) = match res {
                Ok(res) => res,
                Err(LoadError::IOError(err)) => {
                    panic!("error loading {}: {}", path.display(), err);
                }
                Err(LoadError::BrokenRedirect(target, _)) => {
                    *errors = true;
                    println!(
                        "{}:{}: broken redirect to {}",
                        pretty_file.display(),
                        i + 1,
                        target.display()
                    );
                    return;
                }
                Err(LoadError::IsRedirect) => unreachable!(),
            };

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

                let entry = &mut cache.get_mut(&pretty_path).unwrap();
                entry.parse_ids(&pretty_path, &contents, errors);

                if !entry.ids.contains(*fragment) && !is_exception(file, &format!("#{}", fragment))
                {
                    *errors = true;
                    print!("{}:{}: broken link fragment ", pretty_file.display(), i + 1);
                    println!("`#{}` pointing to `{}`", fragment, pretty_path.display());
                };
            }
        } else {
            let pretty_path = path.strip_prefix(root).unwrap_or(&path);
            if !is_exception(file, pretty_path.to_str().unwrap()) {
                *errors = true;
                print!("{}:{}: broken link - ", pretty_file.display(), i + 1);
                println!("{}", pretty_path.display());
            }
        }
    });

    // Search for intra-doc links that rustdoc didn't warn about
    // FIXME(#77199, 77200) Rustdoc should just warn about these directly.
    // NOTE: only looks at one line at a time; in practice this should find most links
    for (i, line) in contents.lines().enumerate() {
        for broken_link in BROKEN_INTRA_DOC_LINK.captures_iter(line) {
            if !is_intra_doc_exception(file, &broken_link[1]) {
                *errors = true;
                print!("{}:{}: broken intra-doc link - ", pretty_file.display(), i + 1);
                println!("{}", &broken_link[0]);
            }
        }
    }
    Some(pretty_file)
}

fn load_file(
    cache: &mut Cache,
    root: &Path,
    file: &Path,
    redirect: Redirect,
) -> Result<(PathBuf, Rc<String>), LoadError> {
    let pretty_file = PathBuf::from(file.strip_prefix(root).unwrap_or(&file));

    let (maybe_redirect, contents) = match cache.entry(pretty_file.clone()) {
        Entry::Occupied(entry) => (None, entry.get().source.clone()),
        Entry::Vacant(entry) => {
            let contents = match fs::read_to_string(file) {
                Ok(s) => Rc::new(s),
                Err(err) => {
                    return Err(if let FromRedirect(true) = redirect {
                        LoadError::BrokenRedirect(file.to_path_buf(), err)
                    } else {
                        LoadError::IOError(err)
                    });
                }
            };

            let maybe = maybe_redirect(&contents);
            if maybe.is_some() {
                if let SkipRedirect = redirect {
                    return Err(LoadError::IsRedirect);
                }
            } else {
                entry.insert(FileEntry { source: contents.clone(), ids: HashSet::new() });
            }
            (maybe, contents)
        }
    };
    match maybe_redirect.map(|url| file.parent().unwrap().join(url)) {
        Some(redirect_file) => load_file(cache, root, &redirect_file, FromRedirect(true)),
        None => Ok((pretty_file, contents)),
    }
}

fn maybe_redirect(source: &str) -> Option<String> {
    const REDIRECT: &str = "<p>Redirecting to <a href=";

    let mut lines = source.lines();
    let redirect_line = lines.nth(6)?;

    redirect_line.find(REDIRECT).map(|i| {
        let rest = &redirect_line[(i + REDIRECT.len() + 1)..];
        let pos_quote = rest.find('"').unwrap();
        rest[..pos_quote].to_owned()
    })
}

fn with_attrs_in_source<F: FnMut(&str, usize, &str)>(contents: &str, attr: &str, mut f: F) {
    let mut base = "";
    for (i, mut line) in contents.lines().enumerate() {
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
