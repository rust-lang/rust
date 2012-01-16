/*
Module: fs

File system manipulation
*/

import core::ctypes;
import core::vec;
import core::option;
import os;
import os::getcwd;
import os_fs;

#[abi = "cdecl"]
native mod rustrt {
    fn rust_path_is_dir(path: str::sbuf) -> ctypes::c_int;
    fn rust_path_exists(path: str::sbuf) -> ctypes::c_int;
}

/*
Function: path_sep

Get the default path separator for the host platform
*/
fn path_sep() -> str { ret str::from_char(os_fs::path_sep); }

// FIXME: This type should probably be constrained
/*
Type: path

A path or fragment of a filesystem path
*/
type path = str;

/*
Function: dirname

Get the directory portion of a path

Returns all of the path up to, but excluding, the final path separator.
The dirname of "/usr/share" will be "/usr", but the dirname of
"/usr/share/" is "/usr/share".

If the path is not prefixed with a directory, then "." is returned.
*/
fn dirname(p: path) -> path {
    let i: int = str::rindex(p, os_fs::path_sep as u8);
    if i == -1 {
        i = str::rindex(p, os_fs::alt_path_sep as u8);
        if i == -1 { ret "."; }
    }
    ret str::substr(p, 0u, i as uint);
}

/*
Function: basename

Get the file name portion of a path

Returns the portion of the path after the final path separator.
The basename of "/usr/share" will be "share". If there are no
path separators in the path then the returned path is identical to
the provided path. If an empty path is provided or the path ends
with a path separator then an empty path is returned.
*/
fn basename(p: path) -> path {
    let i: int = str::rindex(p, os_fs::path_sep as u8);
    if i == -1 {
        i = str::rindex(p, os_fs::alt_path_sep as u8);
        if i == -1 { ret p; }
    }
    let len = str::byte_len(p);
    if i + 1 as uint >= len { ret p; }
    ret str::slice(p, i + 1 as uint, len);
}


// FIXME: Need some typestate to avoid bounds check when len(pre) == 0
/*
Function: connect

Connects to path segments

Given paths `pre` and `post, removes any trailing path separator on `pre` and
any leading path separator on `post`, and returns the concatenation of the two
with a single path separator between them.
*/

fn connect(pre: path, post: path) -> path {
    let pre_ = pre;
    let post_ = post;
    let sep = os_fs::path_sep as u8;
    let pre_len = str::byte_len(pre);
    let post_len = str::byte_len(post);
    if pre_len > 1u && pre[pre_len-1u] == sep { str::pop_byte(pre_); }
    if post_len > 1u && post[0] == sep { str::shift_byte(post_); }
    ret pre_ + path_sep() + post_;
}

/*
Function: connect_many

Connects a vector of path segments into a single path.

Inserts path separators as needed.
*/
fn connect_many(paths: [path]) : vec::is_not_empty(paths) -> path {
    ret if vec::len(paths) == 1u {
        paths[0]
    } else {
        let rest = vec::slice(paths, 1u, vec::len(paths));
        check vec::is_not_empty(rest);
        connect(paths[0], connect_many(rest))
    }
}

/*
Function: path_is_dir

Indicates whether a path represents a directory.
*/
fn path_is_dir(p: path) -> bool {
    ret str::as_buf(p, {|buf| rustrt::rust_path_is_dir(buf) != 0i32 });
}

/*
Function: path_exists

Indicates whether a path exists.
*/
fn path_exists(p: path) -> bool {
    ret str::as_buf(p, {|buf| rustrt::rust_path_exists(buf) != 0i32 });
}

/*
Function: make_dir

Creates a directory at the specified path.
*/
fn make_dir(p: path, mode: ctypes::c_int) -> bool {
    ret mkdir(p, mode);

    #[cfg(target_os = "win32")]
    fn mkdir(_p: path, _mode: ctypes::c_int) -> bool unsafe {
        // FIXME: turn mode into something useful?
        ret str::as_buf(_p, {|buf|
            os::kernel32::CreateDirectoryA(
                buf, unsafe::reinterpret_cast(0))
        });
    }

    #[cfg(target_os = "linux")]
    #[cfg(target_os = "macos")]
    #[cfg(target_os = "freebsd")]
    fn mkdir(_p: path, _mode: ctypes::c_int) -> bool {
        ret str::as_buf(_p, {|buf| os::libc::mkdir(buf, _mode) == 0i32 });
    }
}

/*
Function: list_dir

Lists the contents of a directory.
*/
fn list_dir(p: path) -> [str] {
    let p = p;
    let pl = str::byte_len(p);
    if pl == 0u || p[pl - 1u] as char != os_fs::path_sep { p += path_sep(); }
    let full_paths: [str] = [];
    for filename: str in os_fs::list_dir(p) {
        if !str::eq(filename, ".") {
            if !str::eq(filename, "..") { full_paths += [p + filename]; }
        }
    }
    ret full_paths;
}

/*
Function: remove_dir

Removes a directory at the specified path.
*/
fn remove_dir(p: path) -> bool {
   ret rmdir(p);

    #[cfg(target_os = "win32")]
    fn rmdir(_p: path) -> bool {
        ret str::as_buf(_p, {|buf| os::kernel32::RemoveDirectoryA(buf)});
    }

    #[cfg(target_os = "linux")]
    #[cfg(target_os = "macos")]
    #[cfg(target_os = "freebsd")]
    fn rmdir(_p: path) -> bool {
        ret str::as_buf(_p, {|buf| os::libc::rmdir(buf) == 0i32 });
    }
}

fn change_dir(p: path) -> bool {
    ret chdir(p);

    #[cfg(target_os = "win32")]
    fn chdir(_p: path) -> bool {
        ret str::as_buf(_p, {|buf| os::kernel32::SetCurrentDirectoryA(buf)});
    }

    #[cfg(target_os = "linux")]
    #[cfg(target_os = "macos")]
    #[cfg(target_os = "freebsd")]
    fn chdir(_p: path) -> bool {
        ret str::as_buf(_p, {|buf| os::libc::chdir(buf) == 0i32 });
    }
}

/*
Function: path_is_absolute

Indicates whether a path is absolute.

A path is considered absolute if it begins at the filesystem root ("/") or,
on Windows, begins with a drive letter.
*/
fn path_is_absolute(p: path) -> bool { ret os_fs::path_is_absolute(p); }

// FIXME: under Windows, we should prepend the current drive letter to paths
// that start with a slash.
/*
Function: make_absolute

Convert a relative path to an absolute path

If the given path is relative, return it prepended with the current working
directory. If the given path is already an absolute path, return it
as is.
*/
fn make_absolute(p: path) -> path {
    if path_is_absolute(p) { ret p; } else { ret connect(getcwd(), p); }
}

/*
Function: split

Split a path into it's individual components

Splits a given path by path separators and returns a vector containing
each piece of the path. On Windows, if the path is absolute then
the first element of the returned vector will be the drive letter
followed by a colon.
*/
fn split(p: path) -> [path] {
    let split1 = str::split(p, os_fs::path_sep as u8);
    let split2 = [];
    for s in split1 {
        split2 += str::split(s, os_fs::alt_path_sep as u8);
    }
    ret split2;
}

/*
Function: splitext

Split a path into a pair of strings with the first element being the filename
without the extension and the second being either empty or the file extension
including the period. Leading periods in the basename are ignored.  If the
path includes directory components then they are included in the filename part
of the result pair.
*/
fn splitext(p: path) -> (str, str) {
    if str::is_empty(p) { ("", "") }
    else {
        let parts = str::split(p, '.' as u8);
        if vec::len(parts) > 1u {
            let base = str::connect(vec::init(parts), ".");
            let ext = "." + option::get(vec::last(parts));

            fn is_dotfile(base: str) -> bool {
                str::is_empty(base)
                    || str::ends_with(
                        base, str::from_char(os_fs::path_sep))
                    || str::ends_with(
                        base, str::from_char(os_fs::alt_path_sep))
            }

            fn ext_contains_sep(ext: str) -> bool {
                vec::len(split(ext)) > 1u
            }

            fn no_basename(ext: str) -> bool {
                str::ends_with(
                    ext, str::from_char(os_fs::path_sep))
                    || str::ends_with(
                        ext, str::from_char(os_fs::alt_path_sep))
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

/*
Function: normalize

Removes extra "." and ".." entries from paths.

Does not follow symbolic links.
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

    let s = if str::byte_len(s) == 0u {
        "."
    } else {
        s
    };

    ret s;

    fn strip_dots(s: [path]) -> [path] {
        vec::filter_map(s, { |elem|
            if elem == "." {
                option::none
            } else {
                option::some(elem)
            }
        })
    }

    fn rollup_doubledots(s: [path]) -> [path] {
        if vec::is_empty(s) {
            ret [];
        }

        let t = [];
        let i = vec::len(s);
        let skip = 0;
        do {
            i -= 1u;
            if s[i] == ".." {
                skip += 1;
            } else {
                if skip == 0 {
                    t += [s[i]];
                } else {
                    skip -= 1;
                }
            }
        } while i != 0u;
        let t = vec::reversed(t);
        while skip > 0 {
            t += [".."];
            skip -= 1;
        }
        ret t;
    }

    #[cfg(target_os = "linux")]
    #[cfg(target_os = "macos")]
    #[cfg(target_os = "freebsd")]
    fn reabsolute(orig: path, new: path) -> path {
        if path_is_absolute(orig) {
            path_sep() + new
        } else {
            new
        }
    }

    #[cfg(target_os = "win32")]
    fn reabsolute(orig: path, new: path) -> path {
       if path_is_absolute(orig) && orig[0] == os_fs::path_sep as u8 {
           str::from_char(os_fs::path_sep) + new
       } else {
           new
       }
    }

    fn reterminate(orig: path, new: path) -> path {
        let last = orig[str::byte_len(orig) - 1u];
        if last == os_fs::path_sep as u8
            || last == os_fs::path_sep as u8 {
            ret new + path_sep();
        } else {
            ret new;
        }
    }
}

/*
Function: homedir

Returns the path to the user's home directory, if known.

On Unix, returns the value of the "HOME" environment variable if it is set and
not equal to the empty string.

On Windows, returns the value of the "HOME" environment variable if it is set
and not equal to the empty string. Otherwise, returns the value of the
"USERPROFILE" environment variable if it is set and not equal to the empty
string.

Otherwise, homedir returns option::none.
*/
fn homedir() -> option<path> {
    ret alt generic_os::getenv("HOME") {
        some(p) {
            if !str::is_empty(p) {
                some(p)
            } else {
                secondary()
            }
        }
        none. {
            secondary()
        }
    };

    #[cfg(target_os = "linux")]
    #[cfg(target_os = "macos")]
    #[cfg(target_os = "freebsd")]
    fn secondary() -> option<path> {
        none
    }

    #[cfg(target_os = "win32")]
    fn secondary() -> option<path> {
        option::maybe(none, generic_os::getenv("USERPROFILE")) {|p|
            if !str::is_empty(p) {
                some(p)
            } else {
                none
            }
        }
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn test_connect() {
        let slash = fs::path_sep();
        log(error, fs::connect("a", "b"));
        assert (fs::connect("a", "b") == "a" + slash + "b");
        assert (fs::connect("a" + slash, "b") == "a" + slash + "b");
    }

    // Issue #712
    #[test]
    fn test_list_dir_no_invalid_memory_access() { fs::list_dir("."); }

    #[test]
    fn list_dir() {
        let dirs = fs::list_dir(".");
        // Just assuming that we've got some contents in the current directory
        assert (vec::len(dirs) > 0u);

        for dir in dirs { log(debug, dir); }
    }

    #[test]
    fn path_is_dir() {
        assert (fs::path_is_dir("."));
        assert (!fs::path_is_dir("test/stdtest/fs.rs"));
    }

    #[test]
    fn path_exists() {
        assert (fs::path_exists("."));
        assert (!fs::path_exists("test/nonexistent-bogus-path"));
    }

    fn ps() -> str {
        fs::path_sep()
    }

    fn aps() -> str {
        "/"
    }

    #[test]
    fn split1() {
        let actual = fs::split("a" + ps() + "b");
        let expected = ["a", "b"];
        assert actual == expected;
    }

    #[test]
    fn split2() {
        let actual = fs::split("a" + aps() + "b");
        let expected = ["a", "b"];
        assert actual == expected;
    }

    #[test]
    fn split3() {
        let actual = fs::split(ps() + "a" + ps() + "b");
        let expected = ["a", "b"];
        assert actual == expected;
    }

    #[test]
    fn split4() {
        let actual = fs::split("a" + ps() + "b" + aps() + "c");
        let expected = ["a", "b", "c"];
        assert actual == expected;
    }

    #[test]
    fn normalize1() {
        let actual = fs::normalize("a/b/..");
        let expected = "a";
        assert actual == expected;
    }

    #[test]
    fn normalize2() {
        let actual = fs::normalize("/a/b/..");
        let expected = "/a";
        assert actual == expected;
    }

    #[test]
    fn normalize3() {
        let actual = fs::normalize("a/../b");
        let expected = "b";
        assert actual == expected;
    }

    #[test]
    fn normalize4() {
        let actual = fs::normalize("/a/../b");
        let expected = "/b";
        assert actual == expected;
    }

    #[test]
    fn normalize5() {
        let actual = fs::normalize("a/.");
        let expected = "a";
        assert actual == expected;
    }

    #[test]
    fn normalize6() {
        let actual = fs::normalize("a/./b/");
        let expected = "a/b/";
        assert actual == expected;
    }

    #[test]
    fn normalize7() {
        let actual = fs::normalize("a/..");
        let expected = ".";
        assert actual == expected;
    }

    #[test]
    fn normalize8() {
        let actual = fs::normalize("../../..");
        let expected = "../../..";
        assert actual == expected;
    }

    #[test]
    fn normalize9() {
        let actual = fs::normalize("a/b/../../..");
        let expected = "..";
        assert actual == expected;
    }

    #[test]
    fn normalize10() {
        let actual = fs::normalize("/a/b/c/../d/./../../e/");
        let expected = "/a/e/";
        log(error, actual);
        assert actual == expected;
    }

    #[test]
    fn normalize11() {
        let actual = fs::normalize("/a/..");
        let expected = "/";
        assert actual == expected;
    }

    #[test]
    #[cfg(target_os = "win32")]
    fn normalize12() {
        let actual = fs::normalize("C:/whatever");
        let expected = "C:/whatever";
        log(error, actual);
        assert actual == expected;
    }

    #[test]
    #[cfg(target_os = "win32")]
    fn path_is_absolute_win32() {
        assert fs::path_is_absolute("C:/whatever");
    }

    #[test]
    fn splitext_empty() {
        let (base, ext) = fs::splitext("");
        assert base == "";
        assert ext == "";
    }

    #[test]
    fn splitext_ext() {
        let (base, ext) = fs::splitext("grum.exe");
        assert base == "grum";
        assert ext == ".exe";
    }

    #[test]
    fn splitext_noext() {
        let (base, ext) = fs::splitext("grum");
        assert base == "grum";
        assert ext == "";
    }

    #[test]
    fn splitext_dotfile() {
        let (base, ext) = fs::splitext(".grum");
        assert base == ".grum";
        assert ext == "";
    }

    #[test]
    fn splitext_path_ext() {
        let (base, ext) = fs::splitext("oh/grum.exe");
        assert base == "oh/grum";
        assert ext == ".exe";
    }

    #[test]
    fn splitext_path_noext() {
        let (base, ext) = fs::splitext("oh/grum");
        assert base == "oh/grum";
        assert ext == "";
    }

    #[test]
    fn splitext_dot_in_path() {
        let (base, ext) = fs::splitext("oh.my/grum");
        assert base == "oh.my/grum";
        assert ext == "";
    }

    #[test]
    fn splitext_nobasename() {
        let (base, ext) = fs::splitext("oh.my/");
        assert base == "oh.my/";
        assert ext == "";
    }

    #[test]
    #[cfg(target_os = "linux")]
    #[cfg(target_os = "macos")]
    #[cfg(target_os = "freebsd")]
    fn homedir() {
        import getenv = generic_os::getenv;
        import setenv = generic_os::setenv;

        let oldhome = getenv("HOME");

        setenv("HOME", "/home/MountainView");
        assert fs::homedir() == some("/home/MountainView");

        setenv("HOME", "");
        assert fs::homedir() == none;

        option::may(oldhome, {|s| setenv("HOME", s)});
    }

    #[test]
    #[cfg(target_os = "win32")]
    fn homedir() {
        import getenv = generic_os::getenv;
        import setenv = generic_os::setenv;

        let oldhome = getenv("HOME");
        let olduserprofile = getenv("USERPROFILE");

        setenv("HOME", "");
        setenv("USERPROFILE", "");

        assert fs::homedir() == none;

        setenv("HOME", "/home/MountainView");
        assert fs::homedir() == some("/home/MountainView");

        setenv("HOME", "");

        setenv("USERPROFILE", "/home/MountainView");
        assert fs::homedir() == some("/home/MountainView");

        setenv("USERPROFILE", "/home/MountainView");
        assert fs::homedir() == some("/home/MountainView");

        setenv("HOME", "/home/MountainView");
        setenv("USERPROFILE", "/home/PaloAlto");
        assert fs::homedir() == some("/home/MountainView");

        option::may(oldhome, {|s| setenv("HOME", s)});
        option::may(olduserprofile, {|s| setenv("USERPROFILE", s)});
    }
}


#[test]
fn test() {
    assert (!fs::path_is_absolute("test-path"));

    log(debug, "Current working directory: " + os::getcwd());

    log(debug, fs::make_absolute("test-path"));
    log(debug, fs::make_absolute("/usr/bin"));
}


// Local Variables:
// mode: rust;
// fill-column: 78;
// indent-tabs-mode: nil
// c-basic-offset: 4
// buffer-file-coding-system: utf-8-unix
// End:
