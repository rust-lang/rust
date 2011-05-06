fn getenv(str n) -> str {
    ret Str.str_from_cstr(OS.libc.getenv(Str.buf(n)));
}

