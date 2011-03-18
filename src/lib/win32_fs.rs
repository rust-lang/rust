native "rust" mod rustrt {
  fn rust_list_files(str path) -> vec[str];
  fn rust_file_is_dir(str path) -> int;
}

impure fn list_dir(str path) -> vec[str] {
  ret rustrt.rust_list_files(path+"*");
}

const char path_sep = '\\';

// Local Variables:
// mode: rust;
// fill-column: 78;
// indent-tabs-mode: nil
// c-basic-offset: 4
// buffer-file-coding-system: utf-8-unix
// compile-command: "make -k -C .. 2>&1 | sed -e 's/\\/x\\//x:\\//g'";
// End:
