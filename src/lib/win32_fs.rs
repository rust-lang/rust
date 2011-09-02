

native "rust" mod rustrt {
    fn rust_list_files(path: &str) -> [str];
    fn rust_file_is_dir(path: str) -> int;
}

fn list_dir(path: &str) -> [str] {
    let path = path + "*";
    ret rustrt::rust_list_files(path);
}

fn path_is_absolute(p: &str) -> bool {
    ret str::char_at(p, 0u) == '/' ||
            str::char_at(p, 1u) == ':' && str::char_at(p, 2u) == '\\';
}

/* FIXME: win32 path handling actually accepts '/' or '\' and has subtly
 * different semantics for each. Since we build on mingw, we are usually
 * dealing with /-separated paths. But the whole interface to splitting and
 * joining pathnames needs a bit more abstraction on win32. Possibly a vec or
 * tag type.
 */
const path_sep: char = '/';

const alt_path_sep: char = '\\';
// Local Variables:
// mode: rust;
// fill-column: 78;
// indent-tabs-mode: nil
// c-basic-offset: 4
// buffer-file-coding-system: utf-8-unix
// compile-command: "make -k -C $RBUILD 2>&1 | sed -e 's/\\/x\\//x:\\//g'";
// End:
