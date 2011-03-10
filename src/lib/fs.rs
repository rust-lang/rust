native "rust" mod rustrt {
  fn rust_file_is_dir(str path) -> int;
}

impure fn file_is_dir(str path) -> bool {
  ret rustrt.rust_file_is_dir(path) != 0;
}

impure fn list_dir(str path) -> vec[str] {
  if (path.(_str.byte_len(path)-1u) as char != os_fs.path_sep) {
    path += _str.unsafe_from_bytes(vec(os_fs.path_sep as u8));
  }
  let vec[str] full_paths = vec();
  for (str filename in os_fs.list_dir(path)) {
    if (!_str.eq(filename, ".")) {if (!_str.eq(filename, "..")) {
      full_paths = _vec.push[str](full_paths, path + filename);
    }}
  }
  ret full_paths;
}
