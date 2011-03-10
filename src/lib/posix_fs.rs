native "rust" mod rustrt {
  fn rust_dirent_filename(os.libc.dirent ent) -> str;
}

impure fn list_dir(str path) -> vec[str] {
  // TODO ensure this is always closed
  auto dir = os.libc.opendir(_str.buf(path));
  check (dir as uint != 0u);
  let vec[str] result = vec();
  while (true) {
    auto ent = os.libc.readdir(dir);
    if (ent as int == 0) {break;}
    result = _vec.push[str](result, rustrt.rust_dirent_filename(ent));
  }
  os.libc.closedir(dir);
  ret result;
}

const char path_sep = '/';
