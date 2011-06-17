

native "rust" mod rustrt {
    fn rust_list_files(str path) -> vec[str];
    fn rust_file_is_dir(str path) -> int;
}

fn list_dir(str path) -> vec[str] { ret rustrt::rust_list_files(path + "*"); }

/* FIXME: win32 path handling actually accepts '/' or '\' and has subtly
 * different semantics for each. Since we build on mingw, we are usually
 * dealing with /-separated paths. But the whole interface to splitting and
 * joining pathnames needs a bit more abstraction on win32. Possibly a vec or
 * tag type.
 */
const char path_sep = '/';

const char alt_path_sep = '\\';
// Local Variables:
// mode: rust;
// fill-column: 78;
// indent-tabs-mode: nil
// c-basic-offset: 4
// buffer-file-coding-system: utf-8-unix
// compile-command: "make -k -C $RBUILD 2>&1 | sed -e 's/\\/x\\//x:\\//g'";
// End:
