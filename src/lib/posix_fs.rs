
native "c-stack-cdecl" mod rustrt {
    fn rust_list_files(&&path: str) -> [str];
}

fn list_dir(path: str) -> [str] {
    ret rustrt::rust_list_files(path);

    // FIXME: No idea why, but this appears to corrupt memory on OSX. I
    // suspect it has to do with the tasking primitives somehow, or perhaps
    // the FFI. Worth investigating more when we're digging into the FFI and
    // unsafe mode in more detail; in the meantime we just call list_files
    // above and skip this code.

    /*
    auto dir = os::libc::opendir(str::buf(path));
    assert (dir as uint != 0u);
    let vec<str> result = [];
    while (true) {
        auto ent = os::libc::readdir(dir);
        if (ent as int == 0) {
            os::libc::closedir(dir);
            ret result;
        }
        vec::push::<str>(result, rustrt::rust_dirent_filename(ent));
    }
    os::libc::closedir(dir);
    ret result;
    */

}

fn path_is_absolute(p: str) -> bool { ret str::char_at(p, 0u) == '/'; }

const path_sep: char = '/';

const alt_path_sep: char = '/';

// Local Variables:
// mode: rust;
// fill-column: 78;
// indent-tabs-mode: nil
// c-basic-offset: 4
// buffer-file-coding-system: utf-8-unix
// compile-command: "make -k -C $RBUILD 2>&1 | sed -e 's/\\/x\\//x:\\//g'";
// End:
