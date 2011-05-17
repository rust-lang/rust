fn getenv(str n) -> option::t[str] {
  auto s = os::libc::getenv(str::buf(n));
  if ((s as int) == 0) {
    ret option::none[str];
  } else {
    ret option::some[str](str::str_from_cstr(s));
  }
}

