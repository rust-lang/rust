// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

/*!
 * Support for matching file paths against Unix shell style patterns.
 *
 * The `glob` and `glob_with` functions, in concert with the `Paths`
 * type, allow querying the filesystem for all files that match a particular
 * pattern - just like the libc `glob` function (for an example see the `glob`
 * documentation). The methods on the `Pattern` type provide functionality
 * for checking if individual paths match a particular pattern - in a similar
 * manner to the libc `fnmatch` function
 *
 * For consistency across platforms, and for Windows support, this module
 * is implemented entirely in Rust rather than deferring to the libc
 * `glob`/`fnmatch` functions.
 */

#![crate_id = "glob#0.11.0"]
#![experimental]
#![crate_type = "rlib"]
#![crate_type = "dylib"]
#![license = "MIT/ASL2"]
#![doc(html_logo_url = "http://www.rust-lang.org/logos/rust-logo-128x128-blk-v2.png",
       html_favicon_url = "http://www.rust-lang.org/favicon.ico",
       html_root_url = "http://doc.rust-lang.org/0.11.0/",
       html_playground_url = "http://play.rust-lang.org/")]

use std::cell::Cell;
use std::{cmp, os, path};
use std::io::fs;
use std::path::is_sep;
use std::string::String;

/**
 * An iterator that yields Paths from the filesystem that match a particular
 * pattern - see the `glob` function for more details.
 */
pub struct Paths {
    dir_patterns: Vec<Pattern>,
    require_dir: bool,
    options: MatchOptions,
    todo: Vec<(Path,uint)>,
}

///
/// Return an iterator that produces all the Paths that match the given pattern,
/// which may be absolute or relative to the current working directory.
///
/// This method uses the default match options and is equivalent to calling
/// `glob_with(pattern, MatchOptions::new())`. Use `glob_with` directly if you
/// want to use non-default match options.
///
/// # Example
///
/// Consider a directory `/media/pictures` containing only the files `kittens.jpg`,
/// `puppies.jpg` and `hamsters.gif`:
///
/// ```rust
/// use glob::glob;
///
/// for path in glob("/media/pictures/*.jpg") {
///     println!("{}", path.display());
/// }
/// ```
///
/// The above code will print:
///
/// ```ignore
/// /media/pictures/kittens.jpg
/// /media/pictures/puppies.jpg
/// ```
///
pub fn glob(pattern: &str) -> Paths {
    glob_with(pattern, MatchOptions::new())
}

/**
 * Return an iterator that produces all the Paths that match the given pattern,
 * which may be absolute or relative to the current working directory.
 *
 * This function accepts Unix shell style patterns as described by `Pattern::new(..)`.
 * The options given are passed through unchanged to `Pattern::matches_with(..)` with
 * the exception that `require_literal_separator` is always set to `true` regardless of the
 * value passed to this function.
 *
 * Paths are yielded in alphabetical order, as absolute paths.
 */
pub fn glob_with(pattern: &str, options: MatchOptions) -> Paths {
    #[cfg(windows)]
    fn check_windows_verbatim(p: &Path) -> bool { path::windows::is_verbatim(p) }
    #[cfg(not(windows))]
    fn check_windows_verbatim(_: &Path) -> bool { false }

    // calculate root this way to handle volume-relative Windows paths correctly
    let mut root = os::getcwd();
    let pat_root = Path::new(pattern).root_path();
    if pat_root.is_some() {
        if check_windows_verbatim(pat_root.get_ref()) {
            // FIXME: How do we want to handle verbatim paths? I'm inclined to return nothing,
            // since we can't very well find all UNC shares with a 1-letter server name.
            return Paths {
                dir_patterns: Vec::new(),
                require_dir: false,
                options: options,
                todo: Vec::new(),
            };
        }
        root.push(pat_root.get_ref());
    }

    let root_len = pat_root.map_or(0u, |p| p.as_vec().len());
    let dir_patterns = pattern.slice_from(cmp::min(root_len, pattern.len()))
                       .split_terminator(is_sep)
                       .map(|s| Pattern::new(s))
                       .collect::<Vec<Pattern>>();
    let require_dir = pattern.chars().next_back().map(is_sep) == Some(true);

    let mut todo = Vec::new();
    if dir_patterns.len() > 0 {
        // Shouldn't happen, but we're using -1 as a special index.
        assert!(dir_patterns.len() < -1 as uint);

        fill_todo(&mut todo, dir_patterns.as_slice(), 0, &root, options);
    }

    Paths {
        dir_patterns: dir_patterns,
        require_dir: require_dir,
        options: options,
        todo: todo,
    }
}

impl Iterator<Path> for Paths {

    fn next(&mut self) -> Option<Path> {
        loop {
            if self.dir_patterns.is_empty() || self.todo.is_empty() {
                return None;
            }

            let (path,idx) = self.todo.pop().unwrap();
            // idx -1: was already checked by fill_todo, maybe path was '.' or
            // '..' that we can't match here because of normalization.
            if idx == -1 as uint {
                if self.require_dir && !path.is_dir() { continue; }
                return Some(path);
            }
            let ref pattern = *self.dir_patterns.get(idx);

            if pattern.matches_with(match path.filename_str() {
                // this ugly match needs to go here to avoid a borrowck error
                None => {
                    // FIXME (#9639): How do we handle non-utf8 filenames? Ignore them for now
                    // Ideally we'd still match them against a *
                    continue;
                }
                Some(x) => x
            }, self.options) {
                if idx == self.dir_patterns.len() - 1 {
                    // it is not possible for a pattern to match a directory *AND* its children
                    // so we don't need to check the children

                    if !self.require_dir || path.is_dir() {
                        return Some(path);
                    }
                } else {
                    fill_todo(&mut self.todo, self.dir_patterns.as_slice(),
                              idx + 1, &path, self.options);
                }
            }
        }
    }

}

fn list_dir_sorted(path: &Path) -> Option<Vec<Path>> {
    match fs::readdir(path) {
        Ok(mut children) => {
            children.sort_by(|p1, p2| p2.filename().cmp(&p1.filename()));
            Some(children.move_iter().collect())
        }
        Err(..) => None
    }
}

/**
 * A compiled Unix shell style pattern.
 */
#[deriving(Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Default)]
pub struct Pattern {
    tokens: Vec<PatternToken>,
}

#[deriving(Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
enum PatternToken {
    Char(char),
    AnyChar,
    AnySequence,
    AnyWithin(Vec<CharSpecifier> ),
    AnyExcept(Vec<CharSpecifier> )
}

#[deriving(Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
enum CharSpecifier {
    SingleChar(char),
    CharRange(char, char)
}

#[deriving(PartialEq)]
enum MatchResult {
    Match,
    SubPatternDoesntMatch,
    EntirePatternDoesntMatch
}

impl Pattern {

    /**
     * This function compiles Unix shell style patterns: `?` matches any single
     * character, `*` matches any (possibly empty) sequence of characters and
     * `[...]` matches any character inside the brackets, unless the first
     * character is `!` in which case it matches any character except those
     * between the `!` and the `]`. Character sequences can also specify ranges
     * of characters, as ordered by Unicode, so e.g. `[0-9]` specifies any
     * character between 0 and 9 inclusive.
     *
     * The metacharacters `?`, `*`, `[`, `]` can be matched by using brackets
     * (e.g. `[?]`).  When a `]` occurs immediately following `[` or `[!` then
     * it is interpreted as being part of, rather then ending, the character
     * set, so `]` and NOT `]` can be matched by `[]]` and `[!]]` respectively.
     * The `-` character can be specified inside a character sequence pattern by
     * placing it at the start or the end, e.g. `[abc-]`.
     *
     * When a `[` does not have a closing `]` before the end of the string then
     * the `[` will be treated literally.
     */
    pub fn new(pattern: &str) -> Pattern {

        let chars = pattern.chars().collect::<Vec<_>>();
        let mut tokens = Vec::new();
        let mut i = 0;

        while i < chars.len() {
            match *chars.get(i) {
                '?' => {
                    tokens.push(AnyChar);
                    i += 1;
                }
                '*' => {
                    // *, **, ***, ****, ... are all equivalent
                    while i < chars.len() && *chars.get(i) == '*' {
                        i += 1;
                    }
                    tokens.push(AnySequence);
                }
                '[' => {

                    if i <= chars.len() - 4 && *chars.get(i + 1) == '!' {
                        match chars.slice_from(i + 3).position_elem(&']') {
                            None => (),
                            Some(j) => {
                                let chars = chars.slice(i + 2, i + 3 + j);
                                let cs = parse_char_specifiers(chars);
                                tokens.push(AnyExcept(cs));
                                i += j + 4;
                                continue;
                            }
                        }
                    }
                    else if i <= chars.len() - 3 && *chars.get(i + 1) != '!' {
                        match chars.slice_from(i + 2).position_elem(&']') {
                            None => (),
                            Some(j) => {
                                let cs = parse_char_specifiers(chars.slice(i + 1, i + 2 + j));
                                tokens.push(AnyWithin(cs));
                                i += j + 3;
                                continue;
                            }
                        }
                    }

                    // if we get here then this is not a valid range pattern
                    tokens.push(Char('['));
                    i += 1;
                }
                c => {
                    tokens.push(Char(c));
                    i += 1;
                }
            }
        }

        Pattern { tokens: tokens }
    }

    /**
     * Escape metacharacters within the given string by surrounding them in
     * brackets. The resulting string will, when compiled into a `Pattern`,
     * match the input string and nothing else.
     */
    pub fn escape(s: &str) -> String {
        let mut escaped = String::new();
        for c in s.chars() {
            match c {
                // note that ! does not need escaping because it is only special inside brackets
                '?' | '*' | '[' | ']' => {
                    escaped.push_char('[');
                    escaped.push_char(c);
                    escaped.push_char(']');
                }
                c => {
                    escaped.push_char(c);
                }
            }
        }
        escaped
    }

    /**
     * Return if the given `str` matches this `Pattern` using the default
     * match options (i.e. `MatchOptions::new()`).
     *
     * # Example
     *
     * ```rust
     * use glob::Pattern;
     *
     * assert!(Pattern::new("c?t").matches("cat"));
     * assert!(Pattern::new("k[!e]tteh").matches("kitteh"));
     * assert!(Pattern::new("d*g").matches("doog"));
     * ```
     */
    pub fn matches(&self, str: &str) -> bool {
        self.matches_with(str, MatchOptions::new())
    }

    /**
     * Return if the given `Path`, when converted to a `str`, matches this `Pattern`
     * using the default match options (i.e. `MatchOptions::new()`).
     */
    pub fn matches_path(&self, path: &Path) -> bool {
        // FIXME (#9639): This needs to handle non-utf8 paths
        path.as_str().map_or(false, |s| {
            self.matches(s)
        })
    }

    /**
     * Return if the given `str` matches this `Pattern` using the specified match options.
     */
    pub fn matches_with(&self, str: &str, options: MatchOptions) -> bool {
        self.matches_from(None, str, 0, options) == Match
    }

    /**
     * Return if the given `Path`, when converted to a `str`, matches this `Pattern`
     * using the specified match options.
     */
    pub fn matches_path_with(&self, path: &Path, options: MatchOptions) -> bool {
        // FIXME (#9639): This needs to handle non-utf8 paths
        path.as_str().map_or(false, |s| {
            self.matches_with(s, options)
        })
    }

    fn matches_from(&self,
                    prev_char: Option<char>,
                    mut file: &str,
                    i: uint,
                    options: MatchOptions) -> MatchResult {

        let prev_char = Cell::new(prev_char);

        let require_literal = |c| {
            (options.require_literal_separator && is_sep(c)) ||
            (options.require_literal_leading_dot && c == '.'
             && is_sep(prev_char.get().unwrap_or('/')))
        };

        for (ti, token) in self.tokens.slice_from(i).iter().enumerate() {
            match *token {
                AnySequence => {
                    loop {
                        match self.matches_from(prev_char.get(), file, i + ti + 1, options) {
                            SubPatternDoesntMatch => (), // keep trying
                            m => return m,
                        }

                        if file.is_empty() {
                            return EntirePatternDoesntMatch;
                        }

                        let (some_c, next) = file.slice_shift_char();
                        if require_literal(some_c.unwrap()) {
                            return SubPatternDoesntMatch;
                        }
                        prev_char.set(some_c);
                        file = next;
                    }
                }
                _ => {
                    if file.is_empty() {
                        return EntirePatternDoesntMatch;
                    }

                    let (some_c, next) = file.slice_shift_char();
                    let c = some_c.unwrap();
                    let matches = match *token {
                        AnyChar => {
                            !require_literal(c)
                        }
                        AnyWithin(ref specifiers) => {
                            !require_literal(c) &&
                                in_char_specifiers(specifiers.as_slice(),
                                                   c,
                                                   options)
                        }
                        AnyExcept(ref specifiers) => {
                            !require_literal(c) &&
                                !in_char_specifiers(specifiers.as_slice(),
                                                    c,
                                                    options)
                        }
                        Char(c2) => {
                            chars_eq(c, c2, options.case_sensitive)
                        }
                        AnySequence => {
                            unreachable!()
                        }
                    };
                    if !matches {
                        return SubPatternDoesntMatch;
                    }
                    prev_char.set(some_c);
                    file = next;
                }
            }
        }

        if file.is_empty() {
            Match
        } else {
            SubPatternDoesntMatch
        }
    }

}

// Fills `todo` with paths under `path` to be matched by `patterns[idx]`,
// special-casing patterns to match `.` and `..`, and avoiding `readdir()`
// calls when there are no metacharacters in the pattern.
fn fill_todo(todo: &mut Vec<(Path, uint)>, patterns: &[Pattern], idx: uint, path: &Path,
             options: MatchOptions) {
    // convert a pattern that's just many Char(_) to a string
    fn pattern_as_str(pattern: &Pattern) -> Option<String> {
        let mut s = String::new();
        for token in pattern.tokens.iter() {
            match *token {
                Char(c) => s.push_char(c),
                _ => return None
            }
        }
        return Some(s);
    }

    let add = |todo: &mut Vec<_>, next_path: Path| {
        if idx + 1 == patterns.len() {
            // We know it's good, so don't make the iterator match this path
            // against the pattern again. In particular, it can't match
            // . or .. globs since these never show up as path components.
            todo.push((next_path, -1 as uint));
        } else {
            fill_todo(todo, patterns, idx + 1, &next_path, options);
        }
    };

    let pattern = &patterns[idx];

    match pattern_as_str(pattern) {
        Some(s) => {
            // This pattern component doesn't have any metacharacters, so we
            // don't need to read the current directory to know where to
            // continue. So instead of passing control back to the iterator,
            // we can just check for that one entry and potentially recurse
            // right away.
            let special = "." == s.as_slice() || ".." == s.as_slice();
            let next_path = path.join(s.as_slice());
            if (special && path.is_dir()) || (!special && next_path.exists()) {
                add(todo, next_path);
            }
        },
        None => {
            match list_dir_sorted(path) {
                Some(entries) => {
                    todo.extend(entries.move_iter().map(|x|(x, idx)));

                    // Matching the special directory entries . and .. that refer to
                    // the current and parent directory respectively requires that
                    // the pattern has a leading dot, even if the `MatchOptions` field
                    // `require_literal_leading_dot` is not set.
                    if pattern.tokens.len() > 0 && pattern.tokens.get(0) == &Char('.') {
                        for &special in [".", ".."].iter() {
                            if pattern.matches_with(special, options) {
                                add(todo, path.join(special));
                            }
                        }
                    }
                }
                None => {}
            }
        }
    }
}

fn parse_char_specifiers(s: &[char]) -> Vec<CharSpecifier> {
    let mut cs = Vec::new();
    let mut i = 0;
    while i < s.len() {
        if i + 3 <= s.len() && s[i + 1] == '-' {
            cs.push(CharRange(s[i], s[i + 2]));
            i += 3;
        } else {
            cs.push(SingleChar(s[i]));
            i += 1;
        }
    }
    cs
}

fn in_char_specifiers(specifiers: &[CharSpecifier], c: char, options: MatchOptions) -> bool {

    for &specifier in specifiers.iter() {
        match specifier {
            SingleChar(sc) => {
                if chars_eq(c, sc, options.case_sensitive) {
                    return true;
                }
            }
            CharRange(start, end) => {

                // FIXME: work with non-ascii chars properly (issue #1347)
                if !options.case_sensitive && c.is_ascii() && start.is_ascii() && end.is_ascii() {

                    let start = start.to_ascii().to_lowercase();
                    let end = end.to_ascii().to_lowercase();

                    let start_up = start.to_uppercase();
                    let end_up = end.to_uppercase();

                    // only allow case insensitive matching when
                    // both start and end are within a-z or A-Z
                    if start != start_up && end != end_up {
                        let start = start.to_char();
                        let end = end.to_char();
                        let c = c.to_ascii().to_lowercase().to_char();
                        if c >= start && c <= end {
                            return true;
                        }
                    }
                }

                if c >= start && c <= end {
                    return true;
                }
            }
        }
    }

    false
}

/// A helper function to determine if two chars are (possibly case-insensitively) equal.
fn chars_eq(a: char, b: char, case_sensitive: bool) -> bool {
    if cfg!(windows) && path::windows::is_sep(a) && path::windows::is_sep(b) {
        true
    } else if !case_sensitive && a.is_ascii() && b.is_ascii() {
        // FIXME: work with non-ascii chars properly (issue #1347)
        a.to_ascii().eq_ignore_case(b.to_ascii())
    } else {
        a == b
    }
}

/**
 * Configuration options to modify the behaviour of `Pattern::matches_with(..)`
 */
#[deriving(Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Default)]
pub struct MatchOptions {

    /**
     * Whether or not patterns should be matched in a case-sensitive manner. This
     * currently only considers upper/lower case relationships between ASCII characters,
     * but in future this might be extended to work with Unicode.
     */
    case_sensitive: bool,

    /**
     * If this is true then path-component separator characters (e.g. `/` on Posix)
     * must be matched by a literal `/`, rather than by `*` or `?` or `[...]`
     */
    require_literal_separator: bool,

    /**
     * If this is true then paths that contain components that start with a `.` will
     * not match unless the `.` appears literally in the pattern: `*`, `?` or `[...]`
     * will not match. This is useful because such files are conventionally considered
     * hidden on Unix systems and it might be desirable to skip them when listing files.
     */
    require_literal_leading_dot: bool
}

impl MatchOptions {

    /**
     * Constructs a new `MatchOptions` with default field values. This is used
     * when calling functions that do not take an explicit `MatchOptions` parameter.
     *
     * This function always returns this value:
     *
     * ```rust,ignore
     * MatchOptions {
     *     case_sensitive: true,
     *     require_literal_separator: false.
     *     require_literal_leading_dot: false
     * }
     * ```
     */
    pub fn new() -> MatchOptions {
        MatchOptions {
            case_sensitive: true,
            require_literal_separator: false,
            require_literal_leading_dot: false
        }
    }

}

#[cfg(test)]
mod test {
    use std::os;
    use super::{glob, Pattern, MatchOptions};

    #[test]
    fn test_absolute_pattern() {
        // assume that the filesystem is not empty!
        assert!(glob("/*").next().is_some());
        assert!(glob("//").next().is_some());

        // check windows absolute paths with host/device components
        let root_with_device = os::getcwd().root_path().unwrap().join("*");
        // FIXME (#9639): This needs to handle non-utf8 paths
        assert!(glob(root_with_device.as_str().unwrap()).next().is_some());
    }

    #[test]
    fn test_wildcard_optimizations() {
        assert!(Pattern::new("a*b").matches("a___b"));
        assert!(Pattern::new("a**b").matches("a___b"));
        assert!(Pattern::new("a***b").matches("a___b"));
        assert!(Pattern::new("a*b*c").matches("abc"));
        assert!(!Pattern::new("a*b*c").matches("abcd"));
        assert!(Pattern::new("a*b*c").matches("a_b_c"));
        assert!(Pattern::new("a*b*c").matches("a___b___c"));
        assert!(Pattern::new("abc*abc*abc").matches("abcabcabcabcabcabcabc"));
        assert!(!Pattern::new("abc*abc*abc").matches("abcabcabcabcabcabcabca"));
        assert!(Pattern::new("a*a*a*a*a*a*a*a*a").matches("aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"));
        assert!(Pattern::new("a*b[xyz]c*d").matches("abxcdbxcddd"));
    }

    #[test]
    fn test_lots_of_files() {
        // this is a good test because it touches lots of differently named files
        glob("/*/*/*/*").skip(10000).next();
    }

    #[test]
    fn test_range_pattern() {

        let pat = Pattern::new("a[0-9]b");
        for i in range(0u, 10) {
            assert!(pat.matches(format!("a{}b", i).as_slice()));
        }
        assert!(!pat.matches("a_b"));

        let pat = Pattern::new("a[!0-9]b");
        for i in range(0u, 10) {
            assert!(!pat.matches(format!("a{}b", i).as_slice()));
        }
        assert!(pat.matches("a_b"));

        let pats = ["[a-z123]", "[1a-z23]", "[123a-z]"];
        for &p in pats.iter() {
            let pat = Pattern::new(p);
            for c in "abcdefghijklmnopqrstuvwxyz".chars() {
                assert!(pat.matches(c.to_str().as_slice()));
            }
            for c in "ABCDEFGHIJKLMNOPQRSTUVWXYZ".chars() {
                let options = MatchOptions {case_sensitive: false, .. MatchOptions::new()};
                assert!(pat.matches_with(c.to_str().as_slice(), options));
            }
            assert!(pat.matches("1"));
            assert!(pat.matches("2"));
            assert!(pat.matches("3"));
        }

        let pats = ["[abc-]", "[-abc]", "[a-c-]"];
        for &p in pats.iter() {
            let pat = Pattern::new(p);
            assert!(pat.matches("a"));
            assert!(pat.matches("b"));
            assert!(pat.matches("c"));
            assert!(pat.matches("-"));
            assert!(!pat.matches("d"));
        }

        let pat = Pattern::new("[2-1]");
        assert!(!pat.matches("1"));
        assert!(!pat.matches("2"));

        assert!(Pattern::new("[-]").matches("-"));
        assert!(!Pattern::new("[!-]").matches("-"));
    }

    #[test]
    fn test_unclosed_bracket() {
        // unclosed `[` should be treated literally
        assert!(Pattern::new("abc[def").matches("abc[def"));
        assert!(Pattern::new("abc[!def").matches("abc[!def"));
        assert!(Pattern::new("abc[").matches("abc["));
        assert!(Pattern::new("abc[!").matches("abc[!"));
        assert!(Pattern::new("abc[d").matches("abc[d"));
        assert!(Pattern::new("abc[!d").matches("abc[!d"));
        assert!(Pattern::new("abc[]").matches("abc[]"));
        assert!(Pattern::new("abc[!]").matches("abc[!]"));
    }

    #[test]
    fn test_pattern_matches() {
        let txt_pat = Pattern::new("*hello.txt");
        assert!(txt_pat.matches("hello.txt"));
        assert!(txt_pat.matches("gareth_says_hello.txt"));
        assert!(txt_pat.matches("some/path/to/hello.txt"));
        assert!(txt_pat.matches("some\\path\\to\\hello.txt"));
        assert!(txt_pat.matches("/an/absolute/path/to/hello.txt"));
        assert!(!txt_pat.matches("hello.txt-and-then-some"));
        assert!(!txt_pat.matches("goodbye.txt"));

        let dir_pat = Pattern::new("*some/path/to/hello.txt");
        assert!(dir_pat.matches("some/path/to/hello.txt"));
        assert!(dir_pat.matches("a/bigger/some/path/to/hello.txt"));
        assert!(!dir_pat.matches("some/path/to/hello.txt-and-then-some"));
        assert!(!dir_pat.matches("some/other/path/to/hello.txt"));
    }

    #[test]
    fn test_pattern_escape() {
        let s = "_[_]_?_*_!_";
        assert_eq!(Pattern::escape(s), "_[[]_[]]_[?]_[*]_!_".to_string());
        assert!(Pattern::new(Pattern::escape(s).as_slice()).matches(s));
    }

    #[test]
    fn test_pattern_matches_case_insensitive() {

        let pat = Pattern::new("aBcDeFg");
        let options = MatchOptions {
            case_sensitive: false,
            require_literal_separator: false,
            require_literal_leading_dot: false
        };

        assert!(pat.matches_with("aBcDeFg", options));
        assert!(pat.matches_with("abcdefg", options));
        assert!(pat.matches_with("ABCDEFG", options));
        assert!(pat.matches_with("AbCdEfG", options));
    }

    #[test]
    fn test_pattern_matches_case_insensitive_range() {

        let pat_within = Pattern::new("[a]");
        let pat_except = Pattern::new("[!a]");

        let options_case_insensitive = MatchOptions {
            case_sensitive: false,
            require_literal_separator: false,
            require_literal_leading_dot: false
        };
        let options_case_sensitive = MatchOptions {
            case_sensitive: true,
            require_literal_separator: false,
            require_literal_leading_dot: false
        };

        assert!(pat_within.matches_with("a", options_case_insensitive));
        assert!(pat_within.matches_with("A", options_case_insensitive));
        assert!(!pat_within.matches_with("A", options_case_sensitive));

        assert!(!pat_except.matches_with("a", options_case_insensitive));
        assert!(!pat_except.matches_with("A", options_case_insensitive));
        assert!(pat_except.matches_with("A", options_case_sensitive));
    }

    #[test]
    fn test_pattern_matches_require_literal_separator() {

        let options_require_literal = MatchOptions {
            case_sensitive: true,
            require_literal_separator: true,
            require_literal_leading_dot: false
        };
        let options_not_require_literal = MatchOptions {
            case_sensitive: true,
            require_literal_separator: false,
            require_literal_leading_dot: false
        };

        assert!(Pattern::new("abc/def").matches_with("abc/def", options_require_literal));
        assert!(!Pattern::new("abc?def").matches_with("abc/def", options_require_literal));
        assert!(!Pattern::new("abc*def").matches_with("abc/def", options_require_literal));
        assert!(!Pattern::new("abc[/]def").matches_with("abc/def", options_require_literal));

        assert!(Pattern::new("abc/def").matches_with("abc/def", options_not_require_literal));
        assert!(Pattern::new("abc?def").matches_with("abc/def", options_not_require_literal));
        assert!(Pattern::new("abc*def").matches_with("abc/def", options_not_require_literal));
        assert!(Pattern::new("abc[/]def").matches_with("abc/def", options_not_require_literal));
    }

    #[test]
    fn test_pattern_matches_require_literal_leading_dot() {

        let options_require_literal_leading_dot = MatchOptions {
            case_sensitive: true,
            require_literal_separator: false,
            require_literal_leading_dot: true
        };
        let options_not_require_literal_leading_dot = MatchOptions {
            case_sensitive: true,
            require_literal_separator: false,
            require_literal_leading_dot: false
        };

        let f = |options| Pattern::new("*.txt").matches_with(".hello.txt", options);
        assert!(f(options_not_require_literal_leading_dot));
        assert!(!f(options_require_literal_leading_dot));

        let f = |options| Pattern::new(".*.*").matches_with(".hello.txt", options);
        assert!(f(options_not_require_literal_leading_dot));
        assert!(f(options_require_literal_leading_dot));

        let f = |options| Pattern::new("aaa/bbb/*").matches_with("aaa/bbb/.ccc", options);
        assert!(f(options_not_require_literal_leading_dot));
        assert!(!f(options_require_literal_leading_dot));

        let f = |options| Pattern::new("aaa/bbb/*").matches_with("aaa/bbb/c.c.c.", options);
        assert!(f(options_not_require_literal_leading_dot));
        assert!(f(options_require_literal_leading_dot));

        let f = |options| Pattern::new("aaa/bbb/.*").matches_with("aaa/bbb/.ccc", options);
        assert!(f(options_not_require_literal_leading_dot));
        assert!(f(options_require_literal_leading_dot));

        let f = |options| Pattern::new("aaa/?bbb").matches_with("aaa/.bbb", options);
        assert!(f(options_not_require_literal_leading_dot));
        assert!(!f(options_require_literal_leading_dot));

        let f = |options| Pattern::new("aaa/[.]bbb").matches_with("aaa/.bbb", options);
        assert!(f(options_not_require_literal_leading_dot));
        assert!(!f(options_require_literal_leading_dot));
    }

    #[test]
    fn test_matches_path() {
        // on windows, (Path::new("a/b").as_str().unwrap() == "a\\b"), so this
        // tests that / and \ are considered equivalent on windows
        assert!(Pattern::new("a/b").matches_path(&Path::new("a/b")));
    }
}
