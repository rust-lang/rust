

native "rust" mod rustrt {
    fn rust_list_files(str path) -> vec[str];
    fn rust_dirent_filename(os::libc::dirent ent) -> str;
}

fn list_dir(str path) -> vec[str] {
    ret rustrt::rust_list_files(path);
    // TODO ensure this is always closed

    // FIXME: No idea why, but this appears to corrupt memory on OSX. I
    // suspect it has to do with the tasking primitives somehow, or perhaps
    // the FFI. Worth investigating more when we're digging into the FFI and
    // unsafe mode in more detail; in the meantime we just call list_files
    // above and skip this code.

    /*
    auto dir = os::libc::opendir(str::buf(path));
    assert (dir as uint != 0u);
    let vec[str] result = [];
    while (true) {
        auto ent = os::libc::readdir(dir);
        if (ent as int == 0) {
            os::libc::closedir(dir);
            ret result;
        }
        vec::push[str](result, rustrt::rust_dirent_filename(ent));
    }
    os::libc::closedir(dir);
    ret result;
    */

}

const char path_sep = '/';

const char alt_path_sep = '/';
// Local Variables:
// mode: rust;
// fill-column: 78;
// indent-tabs-mode: nil
// c-basic-offset: 4
// buffer-file-coding-system: utf-8-unix
// compile-command: "make -k -C .. 2>&1 | sed -e 's/\\/x\\//x:\\//g'";
// End:
