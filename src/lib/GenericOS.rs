fn getenv(str n) -> str {
    ret _str.str_from_cstr(os.libc.getenv(_str.buf(n)));
}

