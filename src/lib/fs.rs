
import os::getcwd;
import os_fs;

native "rust" mod rustrt {
    fn rust_file_is_dir(path: str::sbuf) -> int;
}

fn path_sep() -> str { ret str::from_char(os_fs::path_sep); }

type path = str;

fn dirname(p: path) -> path {
    let i: int = str::rindex(p, os_fs::path_sep as u8);
    if i == -1 {
        i = str::rindex(p, os_fs::alt_path_sep as u8);
        if i == -1 { ret "."; }
    }
    ret str::substr(p, 0u, i as uint);
}

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
fn connect(pre: path, post: path) -> path {
    let len = str::byte_len(pre);
    ret if pre[len - 1u] == os_fs::path_sep as u8 {

            // Trailing '/'?
            pre + post
        } else { pre + path_sep() + post };
}

fn connect_many(paths: [path]) : vec::is_not_empty(paths) -> path {
    ret if vec::len(paths) == 1u {
        paths[0]
    } else {
        let rest = vec::slice(paths, 1u, vec::len(paths));
        check vec::is_not_empty(rest);
        connect(paths[0], connect_many(rest))
    }
}

fn file_is_dir(p: path) -> bool {
    ret str::as_buf(p, {|buf| rustrt::rust_file_is_dir(buf) != 0 });
}

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

fn path_is_absolute(p: path) -> bool { ret os_fs::path_is_absolute(p); }

// FIXME: under Windows, we should prepend the current drive letter to paths
// that start with a slash.
fn make_absolute(p: path) -> path {
    if path_is_absolute(p) { ret p; } else { ret connect(getcwd(), p); }
}

fn split(p: path) -> [path] {
    let split1 = str::split(p, os_fs::path_sep as u8);
    let split2 = [];
    for s in split1 {
        split2 += str::split(s, os_fs::alt_path_sep as u8);
    }
    ret split2;
}

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

    fn reabsolute(orig: path, new: path) -> path {
        if path_is_absolute(orig) {
            path_sep() + new
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
// compile-command: "make -k -C $RBUILD 2>&1 | sed -e 's/\\/x\\//x:\\//g'";
// End:
