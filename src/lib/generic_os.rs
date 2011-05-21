fn getenv(str n) -> option::t[str] {
  auto s = os::libc::getenv(str::buf(n));
  ret if ((s as int) == 0) {
    option::none[str]
  } else {
    option::some[str](str::str_from_cstr(s))
  };
}

