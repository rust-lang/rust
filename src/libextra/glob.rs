// Copyright 2013 The Rust Project Developers. See the COPYRIGHT
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
 * The `glob` and `glob_with` functions, in concert with the `GlobIterator`
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

use std::{os, path, util};

use sort;

/**
 * An iterator that yields Paths from the filesystem that match a particular
 * pattern - see the `glob` function for more details.
 */
pub struct GlobIterator {
    priv root: Path,
    priv dir_patterns: ~[Pattern],
    priv options: MatchOptions,
    priv todo: ~[Path]
}

/**
 * Return an iterator that produces all the Paths that match the given pattern,
 * which may be absolute or relative to the current working directory.
 *
 * This method uses the default match options and is equivalent to calling
 * `glob_with(pattern, MatchOptions::new())`. Use `glob_with` directly if you
 * want to use non-default match options.
 *
 * # Example
 *
 * Consider a directory `/media/pictures` containing only the files `kittens.jpg`,
 * `puppies.jpg` and `hamsters.gif`:
 *
 * ~~~ {.rust}
 * for path in glob("/media/pictures/*.jpg") {
 *     println(path.to_str());
 * }
 * ~~~
 *
 * The above code will print:
 *
 * ~~~
 * /media/pictures/kittens.jpg
 * /media/pictures/puppies.jpg
 * ~~~
 */
pub fn glob(pattern: &str) -> GlobIterator {
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
pub fn glob_with(pattern: &str, options: MatchOptions) -> GlobIterator {

    // note that this relies on the glob meta characters not
    // having any special meaning in actual pathnames
    let path = Path(pattern);
    let dir_patterns = path.components.map(|s| Pattern::new(*s));

    let root = if path.is_absolute() {
        Path {components: ~[], .. path} // preserve windows path host/device
    } else {
        os::getcwd()
    };
    let todo = list_dir_sorted(&root);

    GlobIterator {
        root: root,
        dir_patterns: dir_patterns,
        options: options,
        todo: todo,
    }
}

impl Iterator<Path> for GlobIterator {

    fn next(&mut self) -> Option<Path> {
        loop {
            if self.dir_patterns.is_empty() || self.todo.is_empty() {
                return None;
            }

            let path = self.todo.pop();
            let pattern_index = path.components.len() - self.root.components.len() - 1;
            let ref pattern = self.dir_patterns[pattern_index];

            if pattern.matches_with(*path.components.last(), self.options) {

                if pattern_index == self.dir_patterns.len() - 1 {
                    // it is not possible for a pattern to match a directory *AND* its children
                    // so we don't need to check the children
                    return Some(path);
                } else {
                    self.todo.push_all(list_dir_sorted(&path));
                }
            }
        }
    }

}

fn list_dir_sorted(path: &Path) -> ~[Path] {
    let mut children = os::list_dir_path(path);
    sort::quick_sort(children, |p1, p2| p2.components.last() <= p1.components.last());
    children
}

/**
 * A compiled Unix shell style pattern.
 */
#[deriving(Clone, Eq, TotalEq, Ord, TotalOrd, IterBytes, Zero)]
pub struct Pattern {
    priv tokens: ~[PatternToken]
}

#[deriving(Clone, Eq, TotalEq, Ord, TotalOrd, IterBytes)]
enum PatternToken {
    Char(char),
    AnyChar,
    AnySequence,
    AnyWithin(~[CharSpecifier]),
    AnyExcept(~[CharSpecifier])
}

#[deriving(Clone, Eq, TotalEq, Ord, TotalOrd, IterBytes)]
enum CharSpecifier {
    SingleChar(char),
    CharRange(char, char)
}

#[deriving(Eq)]
enum MatchResult {
    Match,
    SubPatternDoesntMatch,
    EntirePatternDoesntMatch
}

impl Pattern {

    /**
     * This function compiles Unix shell style patterns: `?` matches any single character,
     * `*` matches any (possibly empty) sequence of characters and `[...]` matches any character
     * inside the brackets, unless the first character is `!` in which case it matches any
     * character except those between the `!` and the `]`. Character sequences can also specify
     * ranges of characters, as ordered by Unicode, so e.g. `[0-9]` specifies any character
     * between 0 and 9 inclusive.
     *
     * The metacharacters `?`, `*`, `[`, `]` can be matched by using brackets (e.g. `[?]`).
     * When a `]` occurs immediately following `[` or `[!` then it is interpreted as
     * being part of, rather then ending, the character set, so `]` and NOT `]` can be
     * matched by `[]]` and `[!]]` respectively. The `-` character can be specified inside a
     * character sequence pattern by placing it at the start or the end, e.g. `[abc-]`.
     *
     * When a `[` does not have a closing `]` before the end of the string then the `[` will
     * be treated literally.
     */
    pub fn new(pattern: &str) -> Pattern {

        let chars = pattern.iter().to_owned_vec();
        let mut tokens = ~[];
        let mut i = 0;

        while i < chars.len() {
            match chars[i] {
                '?' => {
                    tokens.push(AnyChar);
                    i += 1;
                }
                '*' => {
                    // *, **, ***, ****, ... are all equivalent
                    while i < chars.len() && chars[i] == '*' {
                        i += 1;
                    }
                    tokens.push(AnySequence);
                }
                '[' => {

                    if i <= chars.len() - 4 && chars[i + 1] == '!' {
                        match chars.slice_from(i + 3).position_elem(&']') {
                            None => (),
                            Some(j) => {
                                let cs = parse_char_specifiers(chars.slice(i + 2, i + 3 + j));
                                tokens.push(AnyExcept(cs));
                                i += j + 4;
                                loop;
                            }
                        }
                    }
                    else if i <= chars.len() - 3 && chars[i + 1] != '!' {
                        match chars.slice_from(i + 2).position_elem(&']') {
                            None => (),
                            Some(j) => {
                                let cs = parse_char_specifiers(chars.slice(i + 1, i + 2 + j));
                                tokens.push(AnyWithin(cs));
                                i += j + 3;
                                loop;
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
    pub fn escape(s: &str) -> ~str {
        let mut escaped = ~"";
        for c in s.iter() {
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
     * ~~~ {.rust}
     * assert!(Pattern::new("c?t").matches("cat"));
     * assert!(Pattern::new("k[!e]tteh").matches("kitteh"));
     * assert!(Pattern::new("d*g").matches("doog"));
     * ~~~
     */
    pub fn matches(&self, str: &str) -> bool {
        self.matches_with(str, MatchOptions::new())
    }

    /**
     * Return if the given `Path`, when converted to a `str`, matches this `Pattern`
     * using the default match options (i.e. `MatchOptions::new()`).
     */
    pub fn matches_path(&self, path: &Path) -> bool {
        self.matches(path.to_str())
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
        self.matches_with(path.to_str(), options)
    }

    fn matches_from(&self,
                    mut prev_char: Option<char>,
                    mut file: &str,
                    i: uint,
                    options: MatchOptions) -> MatchResult {

        let require_literal = |c| {
            (options.require_literal_separator && is_sep(c)) ||
            (options.require_literal_leading_dot && c == '.'
             && is_sep(prev_char.unwrap_or_default('/')))
        };

        for (ti, token) in self.tokens.slice_from(i).iter().enumerate() {
            match *token {
                AnySequence => {
                    loop {
                        match self.matches_from(prev_char, file, i + ti + 1, options) {
                            SubPatternDoesntMatch => (), // keep trying
                            m => return m,
                        }

                        if file.is_empty() {
                            return EntirePatternDoesntMatch;
                        }

                        let (c, next) = file.slice_shift_char();
                        if require_literal(c) {
                            return SubPatternDoesntMatch;
                        }
                        prev_char = Some(c);
                        file = next;
                    }
                }
                _ => {
                    if file.is_empty() {
                        return EntirePatternDoesntMatch;
                    }

                    let (c, next) = file.slice_shift_char();
                    let matches = match *token {
                        AnyChar => {
                            !require_literal(c)
                        }
                        AnyWithin(ref specifiers) => {
                            !require_literal(c) && in_char_specifiers(*specifiers, c, options)
                        }
                        AnyExcept(ref specifiers) => {
                            !require_literal(c) && !in_char_specifiers(*specifiers, c, options)
                        }
                        Char(c2) => {
                            chars_eq(c, c2, options.case_sensitive)
                        }
                        AnySequence => {
                            util::unreachable()
                        }
                    };
                    if !matches {
                        return SubPatternDoesntMatch;
                    }
                    prev_char = Some(c);
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

fn parse_char_specifiers(s: &[char]) -> ~[CharSpecifier] {
    let mut cs = ~[];
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

                    let start = start.to_ascii().to_lower();
                    let end = end.to_ascii().to_lower();

                    let start_up = start.to_upper();
                    let end_up = end.to_upper();

                    // only allow case insensitive matching when
                    // both start and end are within a-z or A-Z
                    if start != start_up && end != end_up {
                        let start = start.to_char();
                        let end = end.to_char();
                        let c = c.to_ascii().to_lower().to_char();
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

/// A helper function to determine if a char is a path separator on the current platform.
fn is_sep(c: char) -> bool {
    if cfg!(windows) {
        path::windows::is_sep(c)
    } else {
        path::posix::is_sep(c)
    }
}

/**
 * Configuration options to modify the behaviour of `Pattern::matches_with(..)`
 */
#[deriving(Clone, Eq, TotalEq, Ord, TotalOrd, IterBytes, Zero)]
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
     * ~~~ {.rust}
     * MatchOptions {
     *     case_sensitive: true,
     *     require_literal_separator: false.
     *     require_literal_leading_dot: false
     * }
     * ~~~
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
    use std::{io, os, unstable};
    use std::unstable::finally::Finally;
    use super::*;
    use tempfile;

    #[test]
    fn test_relative_pattern() {

        fn change_then_remove(p: &Path, f: &fn()) {
            do (|| {
                unstable::change_dir_locked(p, || f());
            }).finally {
                os::remove_dir_recursive(p);
            }
        }

        fn mk_file(path: &str, directory: bool) {
            if directory {
                os::make_dir(&Path(path), 0xFFFF);
            } else {
                io::mk_file_writer(&Path(path), [io::Create]);
            }
        }

        fn abs_path(path: &str) -> Path {
            os::getcwd().push_many(Path(path).components)
        }

        fn glob_vec(pattern: &str) -> ~[Path] {
            glob(pattern).collect()
        }

        let root = tempfile::mkdtemp(&os::tmpdir(), "glob-tests");
        let root = root.expect("Should have created a temp directory");

        do change_then_remove(&root) {

            mk_file("aaa", true);
            mk_file("aaa/apple", true);
            mk_file("aaa/orange", true);
            mk_file("aaa/tomato", true);
            mk_file("aaa/tomato/tomato.txt", false);
            mk_file("aaa/tomato/tomoto.txt", false);
            mk_file("bbb", true);
            mk_file("bbb/specials", true);
            mk_file("bbb/specials/!", false);

            // windows does not allow `*` or `?` characters to exist in filenames
            if os::consts::FAMILY != os::consts::windows::FAMILY {
                mk_file("bbb/specials/*", false);
                mk_file("bbb/specials/?", false);
            }

            mk_file("bbb/specials/[", false);
            mk_file("bbb/specials/]", false);
            mk_file("ccc", true);
            mk_file("xyz", true);
            mk_file("xyz/x", false);
            mk_file("xyz/y", false);
            mk_file("xyz/z", false);

            assert_eq!(glob_vec(""), ~[]);
            assert_eq!(glob_vec("."), ~[]);
            assert_eq!(glob_vec(".."), ~[]);

            assert_eq!(glob_vec("aaa"), ~[abs_path("aaa")]);
            assert_eq!(glob_vec("aaa/"), ~[abs_path("aaa")]);
            assert_eq!(glob_vec("a"), ~[]);
            assert_eq!(glob_vec("aa"), ~[]);
            assert_eq!(glob_vec("aaaa"), ~[]);

            assert_eq!(glob_vec("aaa/apple"), ~[abs_path("aaa/apple")]);
            assert_eq!(glob_vec("aaa/apple/nope"), ~[]);

            // windows should support both / and \ as directory separators
            if os::consts::FAMILY == os::consts::windows::FAMILY {
                assert_eq!(glob_vec("aaa\\apple"), ~[abs_path("aaa/apple")]);
            }

            assert_eq!(glob_vec("???/"), ~[
                abs_path("aaa"),
                abs_path("bbb"),
                abs_path("ccc"),
                abs_path("xyz")]);

            assert_eq!(glob_vec("aaa/tomato/tom?to.txt"), ~[
                abs_path("aaa/tomato/tomato.txt"),
                abs_path("aaa/tomato/tomoto.txt")]);

            assert_eq!(glob_vec("xyz/?"), ~[
                abs_path("xyz/x"),
                abs_path("xyz/y"),
                abs_path("xyz/z")]);

            assert_eq!(glob_vec("a*"), ~[abs_path("aaa")]);
            assert_eq!(glob_vec("*a*"), ~[abs_path("aaa")]);
            assert_eq!(glob_vec("a*a"), ~[abs_path("aaa")]);
            assert_eq!(glob_vec("aaa*"), ~[abs_path("aaa")]);
            assert_eq!(glob_vec("*aaa"), ~[abs_path("aaa")]);
            assert_eq!(glob_vec("*aaa*"), ~[abs_path("aaa")]);
            assert_eq!(glob_vec("*a*a*a*"), ~[abs_path("aaa")]);
            assert_eq!(glob_vec("aaa*/"), ~[abs_path("aaa")]);

            assert_eq!(glob_vec("aaa/*"), ~[
                abs_path("aaa/apple"),
                abs_path("aaa/orange"),
                abs_path("aaa/tomato")]);

            assert_eq!(glob_vec("aaa/*a*"), ~[
                abs_path("aaa/apple"),
                abs_path("aaa/orange"),
                abs_path("aaa/tomato")]);

            assert_eq!(glob_vec("*/*/*.txt"), ~[
                abs_path("aaa/tomato/tomato.txt"),
                abs_path("aaa/tomato/tomoto.txt")]);

            assert_eq!(glob_vec("*/*/t[aob]m?to[.]t[!y]t"), ~[
                abs_path("aaa/tomato/tomato.txt"),
                abs_path("aaa/tomato/tomoto.txt")]);

            assert_eq!(glob_vec("aa[a]"), ~[abs_path("aaa")]);
            assert_eq!(glob_vec("aa[abc]"), ~[abs_path("aaa")]);
            assert_eq!(glob_vec("a[bca]a"), ~[abs_path("aaa")]);
            assert_eq!(glob_vec("aa[b]"), ~[]);
            assert_eq!(glob_vec("aa[xyz]"), ~[]);
            assert_eq!(glob_vec("aa[]]"), ~[]);

            assert_eq!(glob_vec("aa[!b]"), ~[abs_path("aaa")]);
            assert_eq!(glob_vec("aa[!bcd]"), ~[abs_path("aaa")]);
            assert_eq!(glob_vec("a[!bcd]a"), ~[abs_path("aaa")]);
            assert_eq!(glob_vec("aa[!a]"), ~[]);
            assert_eq!(glob_vec("aa[!abc]"), ~[]);

            assert_eq!(glob_vec("bbb/specials/[[]"), ~[abs_path("bbb/specials/[")]);
            assert_eq!(glob_vec("bbb/specials/!"), ~[abs_path("bbb/specials/!")]);
            assert_eq!(glob_vec("bbb/specials/[]]"), ~[abs_path("bbb/specials/]")]);

            if os::consts::FAMILY != os::consts::windows::FAMILY {
                assert_eq!(glob_vec("bbb/specials/[*]"), ~[abs_path("bbb/specials/*")]);
                assert_eq!(glob_vec("bbb/specials/[?]"), ~[abs_path("bbb/specials/?")]);
            }

            if os::consts::FAMILY == os::consts::windows::FAMILY {

                assert_eq!(glob_vec("bbb/specials/[![]"), ~[
                    abs_path("bbb/specials/!"),
                    abs_path("bbb/specials/]")]);

                assert_eq!(glob_vec("bbb/specials/[!]]"), ~[
                    abs_path("bbb/specials/!"),
                    abs_path("bbb/specials/[")]);

                assert_eq!(glob_vec("bbb/specials/[!!]"), ~[
                    abs_path("bbb/specials/["),
                    abs_path("bbb/specials/]")]);

            } else {

                assert_eq!(glob_vec("bbb/specials/[![]"), ~[
                    abs_path("bbb/specials/!"),
                    abs_path("bbb/specials/*"),
                    abs_path("bbb/specials/?"),
                    abs_path("bbb/specials/]")]);

                assert_eq!(glob_vec("bbb/specials/[!]]"), ~[
                    abs_path("bbb/specials/!"),
                    abs_path("bbb/specials/*"),
                    abs_path("bbb/specials/?"),
                    abs_path("bbb/specials/[")]);

                assert_eq!(glob_vec("bbb/specials/[!!]"), ~[
                    abs_path("bbb/specials/*"),
                    abs_path("bbb/specials/?"),
                    abs_path("bbb/specials/["),
                    abs_path("bbb/specials/]")]);

                assert_eq!(glob_vec("bbb/specials/[!*]"), ~[
                    abs_path("bbb/specials/!"),
                    abs_path("bbb/specials/?"),
                    abs_path("bbb/specials/["),
                    abs_path("bbb/specials/]")]);

                assert_eq!(glob_vec("bbb/specials/[!?]"), ~[
                    abs_path("bbb/specials/!"),
                    abs_path("bbb/specials/*"),
                    abs_path("bbb/specials/["),
                    abs_path("bbb/specials/]")]);

            }
        };
    }

    #[test]
    fn test_absolute_pattern() {
        // assume that the filesystem is not empty!
        assert!(glob("/*").next().is_some());
        assert!(glob("//").next().is_none());

        // check windows absolute paths with host/device components
        let root_with_device = (Path {components: ~[], .. os::getcwd()}).to_str() + "*";
        assert!(glob(root_with_device).next().is_some());
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
        for i in range(0, 10) {
            assert!(pat.matches(fmt!("a%db", i)));
        }
        assert!(!pat.matches("a_b"));

        let pat = Pattern::new("a[!0-9]b");
        for i in range(0, 10) {
            assert!(!pat.matches(fmt!("a%db", i)));
        }
        assert!(pat.matches("a_b"));

        let pats = ["[a-z123]", "[1a-z23]", "[123a-z]"];
        for &p in pats.iter() {
            let pat = Pattern::new(p);
            for c in "abcdefghijklmnopqrstuvwxyz".iter() {
                assert!(pat.matches(c.to_str()));
            }
            for c in "ABCDEFGHIJKLMNOPQRSTUVWXYZ".iter() {
                let options = MatchOptions {case_sensitive: false, .. MatchOptions::new()};
                assert!(pat.matches_with(c.to_str(), options));
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
        assert_eq!(Pattern::escape(s), ~"_[[]_[]]_[?]_[*]_!_");
        assert!(Pattern::new(Pattern::escape(s)).matches(s));
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
        // on windows, (Path("a/b").to_str() == "a\\b"), so this
        // tests that / and \ are considered equivalent on windows
        assert!(Pattern::new("a/b").matches_path(&Path("a/b")));
    }
}

