import str::sbuf;


#[cfg(target_os = "linux")]
#[cfg(target_os = "macos")]
fn getenv(str n) -> option::t[str] {
    auto s = os::libc::getenv(str::buf(n));
    ret if (s as int == 0) {
            option::none[str]
        } else { option::some[str](str::str_from_cstr(s)) };
}

#[cfg(target_os = "linux")]
#[cfg(target_os = "macos")]
fn setenv(str n, str v) {
    auto nbuf = str::buf(n);
    auto vbuf = str::buf(v);
    os::libc::setenv(nbuf, vbuf, 1);
}

#[cfg(target_os = "win32")]
fn getenv(str n) -> option::t[str]{
    auto nbuf = str::buf(n);
    auto nsize = 256u;
    while (true) {
        auto vstr = str::alloc(nsize - 1u);
        auto vbuf = str::buf(vstr);
        auto res = os::kernel32::GetEnvironmentVariableA(nbuf, vbuf, nsize);
        if (res == 0u) {
            ret option::none;
        } else if (res < nsize) {
            ret option::some(str::str_from_cstr(vbuf));
        } else {
            nsize = res;
        }
    }
    fail;
}

#[cfg(target_os = "win32")]
fn setenv(str n, str v) {
    auto nbuf = str::buf(n);
    auto vbuf = str::buf(v);
    os::kernel32::SetEnvironmentVariableA(nbuf, vbuf);
}

// Local Variables:
// mode: rust;
// fill-column: 78;
// indent-tabs-mode: nil
// c-basic-offset: 4
// buffer-file-coding-system: utf-8-unix
// compile-command: "make -k -C .. 2>&1 | sed -e 's/\\/x\\//x:\\//g'";
// End:
