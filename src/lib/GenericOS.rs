fn getenv(str n) -> Option.t[str] {
  auto s = OS.libc.getenv(Str.buf(n));
  if ((s as int) == 0) {
    ret Option.none[str];
  } else {
    ret Option.some[str](Str.str_from_cstr(s));
  }
}

