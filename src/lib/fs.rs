
import os::getcwd;
import os_fs;
import str;

native "rust" mod rustrt {
    fn rust_file_is_dir(path: istr::sbuf) -> int;
}

fn path_sep() -> istr { ret istr::from_char(os_fs::path_sep); }

type path = istr;

fn dirname(p: &path) -> path {
    let i: int = istr::rindex(p, os_fs::path_sep as u8);
    if i == -1 {
        i = istr::rindex(p, os_fs::alt_path_sep as u8);
        if i == -1 { ret ~"."; }
    }
    ret istr::substr(p, 0u, i as uint);
}

fn basename(p: &path) -> path {
    let i: int = istr::rindex(p, os_fs::path_sep as u8);
    if i == -1 {
        i = istr::rindex(p, os_fs::alt_path_sep as u8);
        if i == -1 { ret p; }
    }
    let len = istr::byte_len(p);
    if i + 1 as uint >= len { ret p; }
    ret istr::slice(p, i + 1 as uint, len);
}


// FIXME: Need some typestate to avoid bounds check when len(pre) == 0
fn connect(pre: &path, post: &path) -> path {
    let len = istr::byte_len(pre);
    ret if pre[len - 1u] == os_fs::path_sep as u8 {

            // Trailing '/'?
            pre + post
        } else { pre + path_sep() + post };
}

fn file_is_dir(p: &path) -> bool {
    ret istr::as_buf(p, { |buf|
        rustrt::rust_file_is_dir(buf) != 0
    });
}

fn list_dir(p: &path) -> [istr] {
    let p = p;
    let pl = istr::byte_len(p);
    if pl == 0u || p[pl - 1u] as char != os_fs::path_sep { p += path_sep(); }
    let full_paths: [istr] = [];
    for filename: istr in os_fs::list_dir(p) {
        if !istr::eq(filename, ~".") {
            if !istr::eq(filename, ~"..") { full_paths += [p + filename]; }
        }
    }
    ret full_paths;
}

fn path_is_absolute(p: &path) -> bool {
    ret os_fs::path_is_absolute(p);
}

// FIXME: under Windows, we should prepend the current drive letter to paths
// that start with a slash.
fn make_absolute(p: &path) -> path {
    if path_is_absolute(p) {
        ret p;
    } else {
        ret connect(getcwd(), p);
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
