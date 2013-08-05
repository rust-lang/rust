// Copyright 2013 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.


use std::{os, uint};

use sort;


/**
 * An iterator that yields Paths from the filesystem that match a particular
 * pattern - see the glob function for more details.
 */
pub struct GlobIterator {
    priv root: Path,
    priv dir_patterns: ~[Pattern],
    priv todo: ~[Path]
}

/**
 * Return an iterator that produces all the Paths that match the given pattern,
 * which may be absolute or relative to the current working directory.
 *
 * This function accepts Unix shell style patterns as described by Pattern::new(..),
 * with the exception that path separators (i.e. '/' on Posix systems) must be matched
 * by their literal representation - they can't be matched by '*', '?', '[...]', etc.
 *
 * Paths are yielded in alphabetical order, as absolute paths.
 */
pub fn glob(pattern: &str) -> GlobIterator {

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

            if self.dir_patterns[pattern_index].matches(*path.components.last()) {

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
pub struct Pattern {
    priv tokens: ~[PatternToken]
}

enum PatternToken {
    Char(char),
    AnyChar,
    AnySequence,
    AnyWithin(~[char]),
    AnyExcept(~[char])
}

impl Pattern {

    /**
     * This function compiles Unix shell style patterns:
     *   '?' matches any single character.
     *   '*' matches any (possibly empty) sequence of characters.
     *   '[...]' matches any character inside the brackets, unless the first character
     *           is '!' in which case it matches any character except those between
     *           the '!' and the ']'.
     *
     * The metacharacters '?', '*', '[', ']' can be matched by using brackets (e.g. '[?]').
     * When a ']' occurs immediately following '[' or '[!' then it is interpreted as
     * being part of, rather then ending, the character set, so ']' and NOT ']' can be
     * matched by '[]]' and '[!]]' respectively.
     *
     * This function will fail if the given pattern string is malformed - e.g. if it contains
     * a '[...]' sequence that is missing its closing ']'.
     */
    pub fn new(pattern: &str) -> Pattern {
        let mut tokens = ~[];

        let mut pattern_iter = pattern.iter();
        loop {
            let pchar = match pattern_iter.next() {
                None => break,
                Some(c) => c,
            };
            match pchar {
                '?' => {
                    tokens.push(AnyChar);
                }
                '*' => {
                    tokens.push(AnySequence);
                }
                '[' => {

                    // get the next char, or fail with a helpful message
                    let next_pattern_char = || {
                        match pattern_iter.next() {
                            None => fail!("invalid pattern syntax due to unclosed bracket: %s",
                                          pattern),
                            Some(c) => c,
                        }
                    };

                    let c = next_pattern_char();
                    let is_except = (c == '!');

                    let mut chars = ~[];
                    if is_except {
                        chars.push(next_pattern_char());
                    } else {
                        chars.push(c);
                    };

                    loop {
                        match next_pattern_char() {
                            ']' => break,
                            c => chars.push(c),
                        }
                    }
                    tokens.push(if is_except { AnyExcept(chars) } else { AnyWithin(chars) });
                }
                c => {
                    tokens.push(Char(c));
                }
            }
        }

        Pattern { tokens: tokens }
    }

    /**
     * Return if the given str matches this Pattern.
     */
    pub fn matches(&self, file: &str) -> bool {
        self.matches_from(file, 0)
    }

    /**
     * Return if the given Path, when converted to a str, matches this Pattern.
     */
    pub fn matches_path(&self, path: &Path) -> bool {
        self.matches(path.to_str())
    }

    fn matches_from(&self, mut file: &str, i: uint) -> bool {

        for uint::range(i, self.tokens.len()) |ti| {
            match self.tokens[ti] {
                AnySequence => {
                    loop {
                        if self.matches_from(file, ti + 1) {
                            return true;
                        }
                        if file.is_empty() {
                            return false;
                        }
                        file = file.slice_shift_char().second();
                    }
                }
                _ => {
                    if file.is_empty() {
                        return false;
                    }
                    let (c, next) = file.slice_shift_char();
                    let matches = match self.tokens[ti] {
                        AnyChar => true,
                        AnyWithin(ref chars) => chars.contains(&c),
                        AnyExcept(ref chars) => !chars.contains(&c),
                        Char(c2) => c == c2,
                        AnySequence => fail!(),
                    };
                    if !matches {
                        return false;
                    }
                    file = next;
                }
            }
        }

        file.is_empty()
    }

}

#[cfg(test)]
mod test {
    use std::{io, os, unstable};
    use super::*;

    #[test]
    fn test_relative_pattern() {

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

        mk_file("tmp", true);
        mk_file("tmp/glob-tests", true);

        do unstable::change_dir_locked(&Path("tmp/glob-tests")) {

            mk_file("aaa", true);
            mk_file("aaa/apple", true);
            mk_file("aaa/orange", true);
            mk_file("aaa/tomato", true);
            mk_file("aaa/tomato/tomato.txt", false);
            mk_file("aaa/tomato/tomoto.txt", false);
            mk_file("bbb", true);
            mk_file("bbb/specials", true);
            mk_file("bbb/specials/!", false);

            // windows does not allow '*' or '?' characters to exist in filenames
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
    fn test_lots_of_files() {
        // this is a good test because it touches lots of differently named files
        for glob("/*/*/*/*").advance |_p| {}
    }

    #[test]
    #[should_fail]
    #[cfg(not(windows))]
    fn test_unclosed_bracket() {
        Pattern::new("abc[def");
    }

    #[test]
    #[should_fail]
    #[cfg(not(windows))]
    fn test_unclosed_bracket_special() {
        Pattern::new("abc[]"); // not valid syntax, '[]]' should be used to match ']'
    }

    #[test]
    #[should_fail]
    #[cfg(not(windows))]
    fn test_unclosed_bracket_special_except() {
        Pattern::new("abc[!]"); // not valid syntax, '[!]]' should be used to match NOT ']'
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
}

