native "rust" mod rustrt {
  fn rust_dirent_filename(OS.libc.dirent ent) -> str;
}

fn list_dir(str path) -> vec[str] {
  // TODO ensure this is always closed
  auto dir = OS.libc.opendir(Str.buf(path));
  assert (dir as uint != 0u);
  let vec[str] result = vec();
  while (true) {
    auto ent = OS.libc.readdir(dir);
    if (ent as int == 0) {
        OS.libc.closedir(dir);
        ret result;
    }
    Vec.push[str](result, rustrt.rust_dirent_filename(ent));
  }
  OS.libc.closedir(dir);
  ret result;
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
