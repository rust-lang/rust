native "rust" mod rustrt {
  fn rust_list_files(str path) -> vec[str];
  fn rust_file_is_dir(str path) -> int;
}

impure fn list_dir(str path) -> vec[str] {
  ret rustrt.rust_list_files(path+"*");
}

const char path_sep = '\\';
