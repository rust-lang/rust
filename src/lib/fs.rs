/*
Module: fs

File system manipulation
*/

import os::getcwd;
import os_fs;

#[abi = "cdecl"]
native mod rustrt {
    fn rust_file_is_dir(path: str::sbuf) -> int;
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

Given paths `pre` and `post` this function will return a path
that is equal to `post` appended to `pre`, inserting a path separator
between the two as needed.
*/
fn connect(pre: path, post: path) -> path {
    let len = str::byte_len(pre);
    ret if pre[len - 1u] == os_fs::path_sep as u8 {

            // Trailing '/'?
            pre + post
        } else { pre + path_sep() + post };
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
Function: file_id_dir

Indicates whether a path represents a directory.
*/
fn file_is_dir(p: path) -> bool {
    ret str::as_buf(p, {|buf| rustrt::rust_file_is_dir(buf) != 0 });
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
        vec::filter_map({ |elem|
            if elem == "." {
                option::none
            } else {
                option::some(elem)
            }
        }, s)
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

// Local Variables:
// mode: rust;
// fill-column: 78;
// indent-tabs-mode: nil
// c-basic-offset: 4
// buffer-file-coding-system: utf-8-unix
// End:
