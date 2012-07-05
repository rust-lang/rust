//! Path data type and helper functions

export path;
export consts;
export path_is_absolute;
export path_sep;
export dirname;
export basename;
export connect;
export connect_many;
export split;
export splitext;
export normalize;

// FIXME: This type should probably be constrained (#2624)
/// A path or fragment of a filesystem path
type path = str;

#[cfg(unix)]
mod consts {
    /**
     * The primary path separator character for the platform
     *
     * On all platforms it is '/'
     */
    const path_sep: char = '/';
    /**
     * The secondary path separator character for the platform
     *
     * On Unixes it is '/'. On Windows it is '\'.
     */
    const alt_path_sep: char = '/';
}

#[cfg(windows)]
mod consts {
    const path_sep: char = '/';
    const alt_path_sep: char = '\\';
}

/**
 * Indicates whether a path is absolute.
 *
 * A path is considered absolute if it begins at the filesystem root ("/") or,
 * on Windows, begins with a drive letter.
 */
#[cfg(unix)]
fn path_is_absolute(p: path) -> bool {
    str::char_at(p, 0u) == '/'
}

#[cfg(windows)]
fn path_is_absolute(p: str) -> bool {
    ret str::char_at(p, 0u) == '/' ||
        str::char_at(p, 1u) == ':'
        && (str::char_at(p, 2u) == consts::path_sep
            || str::char_at(p, 2u) == consts::alt_path_sep);
}

/// Get the default path separator for the host platform
fn path_sep() -> str { ret str::from_char(consts::path_sep); }

fn split_dirname_basename (pp: path) -> {dirname: str, basename: str} {
    alt str::rfind(pp, |ch|
        ch == consts::path_sep || ch == consts::alt_path_sep
    ) {
      some(i) {
        {dirname: str::slice(pp, 0u, i),
         basename: str::slice(pp, i + 1u, str::len(pp))}
      }
      none { {dirname: ".", basename: pp} }
    }
}

/**
 * Get the directory portion of a path
 *
 * Returns all of the path up to, but excluding, the final path separator.
 * The dirname of "/usr/share" will be "/usr", but the dirname of
 * "/usr/share/" is "/usr/share".
 *
 * If the path is not prefixed with a directory, then "." is returned.
 */
fn dirname(pp: path) -> path {
    ret split_dirname_basename(pp).dirname;
}

/**
 * Get the file name portion of a path
 *
 * Returns the portion of the path after the final path separator.
 * The basename of "/usr/share" will be "share". If there are no
 * path separators in the path then the returned path is identical to
 * the provided path. If an empty path is provided or the path ends
 * with a path separator then an empty path is returned.
 */
fn basename(pp: path) -> path {
    ret split_dirname_basename(pp).basename;
}

/**
 * Connects to path segments
 *
 * Given paths `pre` and `post, removes any trailing path separator on `pre`
 * and any leading path separator on `post`, and returns the concatenation of
 * the two with a single path separator between them.
 */
fn connect(pre: path, post: path) -> path {
    let mut pre_ = pre;
    let mut post_ = post;
    let sep = consts::path_sep as u8;
    let pre_len  = str::len(pre);
    let post_len = str::len(post);
    unsafe {
        if pre_len > 1u && pre[pre_len-1u] == sep {
            str::unsafe::pop_byte(pre_);
        }
        if post_len > 1u && post[0] == sep {
            str::unsafe::shift_byte(post_);
        }
    }
    ret pre_ + path_sep() + post_;
}

/**
 * Connects a vector of path segments into a single path.
 *
 * Inserts path separators as needed.
 */
fn connect_many(paths: ~[path]) -> path {
    ret if vec::len(paths) == 1u {
        paths[0]
    } else {
        let rest = vec::slice(paths, 1u, vec::len(paths));
        connect(paths[0], connect_many(rest))
    }
}

/**
 * Split a path into its individual components
 *
 * Splits a given path by path separators and returns a vector containing
 * each piece of the path. On Windows, if the path is absolute then
 * the first element of the returned vector will be the drive letter
 * followed by a colon.
 */
fn split(p: path) -> ~[path] {
    str::split_nonempty(p, |c| {
        c == consts::path_sep || c == consts::alt_path_sep
    })
}

/**
 * Split a path into the part before the extension and the extension
 *
 * Split a path into a pair of strings with the first element being the
 * filename without the extension and the second being either empty or the
 * file extension including the period. Leading periods in the basename are
 * ignored.  If the path includes directory components then they are included
 * in the filename part of the result pair.
 */
fn splitext(p: path) -> (str, str) {
    if str::is_empty(p) { ("", "") }
    else {
        let parts = str::split_char(p, '.');
        if vec::len(parts) > 1u {
            let base = str::connect(vec::init(parts), ".");
            // We just checked that parts is non-empty, so this is safe
            let ext = "." + vec::last(parts);

            fn is_dotfile(base: str) -> bool {
                str::is_empty(base)
                    || str::ends_with(
                        base, str::from_char(consts::path_sep))
                    || str::ends_with(
                        base, str::from_char(consts::alt_path_sep))
            }

            fn ext_contains_sep(ext: str) -> bool {
                vec::len(split(ext)) > 1u
            }

            fn no_basename(ext: str) -> bool {
                str::ends_with(
                    ext, str::from_char(consts::path_sep))
                    || str::ends_with(
                        ext, str::from_char(consts::alt_path_sep))
            }

            if is_dotfile(base)
                || ext_contains_sep(ext)
                || no_basename(ext) {
                (p, "")
            } else {
                (base, ext)
            }
        } else {
            (p, "")
        }
    }
}

/**
 * Collapses redundant path separators.
 *
 * Does not follow symbolic links.
 *
 * # Examples
 *
 * * '/a/../b' becomes '/b'
 * * 'a/./b/' becomes 'a/b/'
 * * 'a/b/../../../' becomes '..'
 * * '/a/b/c/../d/./../../e/' becomes '/a/e/'
 */
fn normalize(p: path) -> path {
    let s = split(p);
    let s = strip_dots(s);
    let s = rollup_doubledots(s);

    let s = if check vec::is_not_empty(s) {
        connect_many(s)
    } else {
        ""
    };
    let s = reabsolute(p, s);
    let s = reterminate(p, s);

    let s = if str::len(s) == 0u {
        "."
    } else {
        s
    };

    ret s;

    fn strip_dots(s: ~[path]) -> ~[path] {
        vec::filter_map(s, |elem|
            if elem == "." {
                option::none
            } else {
                option::some(elem)
            })
    }

    fn rollup_doubledots(s: ~[path]) -> ~[path] {
        if vec::is_empty(s) {
            ret ~[];
        }

        let mut t = ~[];
        let mut i = vec::len(s);
        let mut skip = 0;
        while i != 0u {
            i -= 1u;
            if s[i] == ".." {
                skip += 1;
            } else {
                if skip == 0 {
                    vec::push(t, s[i]);
                } else {
                    skip -= 1;
                }
            }
        }
        let mut t = vec::reversed(t);
        while skip > 0 {
            vec::push(t, "..");
            skip -= 1;
        }
        ret t;
    }

    #[cfg(unix)]
    fn reabsolute(orig: path, n: path) -> path {
        if path_is_absolute(orig) {
            path_sep() + n
        } else {
            n
        }
    }

    #[cfg(windows)]
    fn reabsolute(orig: path, newp: path) -> path {
       if path_is_absolute(orig) && orig[0] == consts::path_sep as u8 {
           str::from_char(consts::path_sep) + newp
       } else {
           newp
       }
    }

    fn reterminate(orig: path, newp: path) -> path {
        let last = orig[str::len(orig) - 1u];
        if last == consts::path_sep as u8
            || last == consts::path_sep as u8 {
            ret newp + path_sep();
        } else {
            ret newp;
        }
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn test_connect() {
        let slash = path_sep();
        log(error, connect("a", "b"));
        assert (connect("a", "b") == "a" + slash + "b");
        assert (connect("a" + slash, "b") == "a" + slash + "b");
    }

    fn ps() -> str {
        path_sep()
    }

    fn aps() -> str {
        "/"
    }

    #[test]
    fn split1() {
        let actual = split("a" + ps() + "b");
        let expected = ~["a", "b"];
        assert actual == expected;
    }

    #[test]
    fn split2() {
        let actual = split("a" + aps() + "b");
        let expected = ~["a", "b"];
        assert actual == expected;
    }

    #[test]
    fn split3() {
        let actual = split(ps() + "a" + ps() + "b");
        let expected = ~["a", "b"];
        assert actual == expected;
    }

    #[test]
    fn split4() {
        let actual = split("a" + ps() + "b" + aps() + "c");
        let expected = ~["a", "b", "c"];
        assert actual == expected;
    }

    #[test]
    fn normalize1() {
        let actual = normalize("a/b/..");
        let expected = "a";
        assert actual == expected;
    }

    #[test]
    fn normalize2() {
        let actual = normalize("/a/b/..");
        let expected = "/a";
        assert actual == expected;
    }

    #[test]
    fn normalize3() {
        let actual = normalize("a/../b");
        let expected = "b";
        assert actual == expected;
    }

    #[test]
    fn normalize4() {
        let actual = normalize("/a/../b");
        let expected = "/b";
        assert actual == expected;
    }

    #[test]
    fn normalize5() {
        let actual = normalize("a/.");
        let expected = "a";
        assert actual == expected;
    }

    #[test]
    fn normalize6() {
        let actual = normalize("a/./b/");
        let expected = "a/b/";
        assert actual == expected;
    }

    #[test]
    fn normalize7() {
        let actual = normalize("a/..");
        let expected = ".";
        assert actual == expected;
    }

    #[test]
    fn normalize8() {
        let actual = normalize("../../..");
        let expected = "../../..";
        assert actual == expected;
    }

    #[test]
    fn normalize9() {
        let actual = normalize("a/b/../../..");
        let expected = "..";
        assert actual == expected;
    }

    #[test]
    fn normalize10() {
        let actual = normalize("/a/b/c/../d/./../../e/");
        let expected = "/a/e/";
        log(error, actual);
        assert actual == expected;
    }

    #[test]
    fn normalize11() {
        let actual = normalize("/a/..");
        let expected = "/";
        assert actual == expected;
    }

    #[test]
    #[cfg(windows)]
    fn normalize12() {
        let actual = normalize("C:/whatever");
        let expected = "C:/whatever";
        log(error, actual);
        assert actual == expected;
    }

    #[test]
    #[cfg(windows)]
    fn path_is_absolute_win32() {
        assert path_is_absolute("C:/whatever");
    }

    #[test]
    fn splitext_empty() {
        let (base, ext) = splitext("");
        assert base == "";
        assert ext == "";
    }

    #[test]
    fn splitext_ext() {
        let (base, ext) = splitext("grum.exe");
        assert base == "grum";
        assert ext == ".exe";
    }

    #[test]
    fn splitext_noext() {
        let (base, ext) = splitext("grum");
        assert base == "grum";
        assert ext == "";
    }

    #[test]
    fn splitext_dotfile() {
        let (base, ext) = splitext(".grum");
        assert base == ".grum";
        assert ext == "";
    }

    #[test]
    fn splitext_path_ext() {
        let (base, ext) = splitext("oh/grum.exe");
        assert base == "oh/grum";
        assert ext == ".exe";
    }

    #[test]
    fn splitext_path_noext() {
        let (base, ext) = splitext("oh/grum");
        assert base == "oh/grum";
        assert ext == "";
    }

    #[test]
    fn splitext_dot_in_path() {
        let (base, ext) = splitext("oh.my/grum");
        assert base == "oh.my/grum";
        assert ext == "";
    }

    #[test]
    fn splitext_nobasename() {
        let (base, ext) = splitext("oh.my/");
        assert base == "oh.my/";
        assert ext == "";
    }
}

// Local Variables:
// mode: rust;
// fill-column: 78;
// indent-tabs-mode: nil
// c-basic-offset: 4
// buffer-file-coding-system: utf-8-unix
// End:
