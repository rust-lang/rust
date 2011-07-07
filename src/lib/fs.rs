
import os::getcwd;
import os_fs;
import str;

native "rust" mod rustrt {
    fn rust_file_is_dir(str path) -> int;
}

fn path_sep() -> str { ret str::from_char(os_fs::path_sep); }

type path = str;

fn dirname(path p) -> path {
    let int i = str::rindex(p, os_fs::path_sep as u8);
    if (i == -1) {
        i = str::rindex(p, os_fs::alt_path_sep as u8);
        if (i == -1) { ret "."; }
    }
    ret str::substr(p, 0u, i as uint);
}

fn basename(path p) -> path {
    let int i = str::rindex(p, os_fs::path_sep as u8);
    if (i == -1) {
        i = str::rindex(p, os_fs::alt_path_sep as u8);
        if (i == -1) { ret p; }
    }
    auto len = str::byte_len(p);
    if (i + 1 as uint >= len) { ret p; }
    ret str::slice(p, i + 1 as uint, len);
}


// FIXME: Need some typestate to avoid bounds check when len(pre) == 0
fn connect(path pre, path post) -> path {
    auto len = str::byte_len(pre);
    ret if (pre.(len - 1u) == os_fs::path_sep as u8) {
            // Trailing '/'?
            pre + post
        } else { pre + path_sep() + post };
}

fn file_is_dir(path p) -> bool { ret rustrt::rust_file_is_dir(p) != 0; }

fn list_dir(path p) -> vec[str] {
    auto pl = str::byte_len(p);
    if (pl == 0u || p.(pl - 1u) as char != os_fs::path_sep) {
        p += path_sep();
    }
    let vec[str] full_paths = [];
    for (str filename in os_fs::list_dir(p)) {
        if (!str::eq(filename, ".")) {
            if (!str::eq(filename, "..")) {
                vec::push[str](full_paths, p + filename);
            }
        }
    }
    ret full_paths;
}

fn path_is_absolute(path p) -> bool {
    ret os_fs::path_is_absolute(p);
}

// FIXME: under Windows, we should prepend the current drive letter to paths
// that start with a slash.
fn make_absolute(path p) -> path {
    if(path_is_absolute(p)) {
        ret p;
    }
    else {
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
