import str::sbuf;


#[cfg(target_os = "linux")]
#[cfg(target_os = "macos")]
fn getenv(n: &istr) -> option::t<istr> {
    let n = istr::to_estr(n);
    let s = os::libc::getenv(str::buf(n));
    ret if s as int == 0 {
        option::none::<istr>
    } else {
        let s = unsafe::reinterpret_cast(s);
        option::some::<istr>(istr::str_from_cstr(s))
    };
}

#[cfg(target_os = "linux")]
#[cfg(target_os = "macos")]
fn setenv(n: &istr, v: &istr) {
    let n = istr::to_estr(n);
    let v = istr::to_estr(v);
    let nbuf = str::buf(n);
    let vbuf = str::buf(v);
    os::libc::setenv(nbuf, vbuf, 1);
}

#[cfg(target_os = "win32")]
fn getenv(n: &istr) -> option::t<istr> {
    let n = istr::to_estr(n);
    let nbuf = str::buf(n);
    let nsize = 256u;
    while true {
        let vstr = str::alloc(nsize - 1u);
        let vbuf = str::buf(vstr);
        let res = os::kernel32::GetEnvironmentVariableA(nbuf, vbuf, nsize);
        if res == 0u {
            ret option::none;
        } else if res < nsize {
            let vbuf = unsafe::reinterpret_cast(vbuf);
            ret option::some(istr::str_from_cstr(vbuf));
        } else { nsize = res; }
    }
    fail;
}

#[cfg(target_os = "win32")]
fn setenv(n: &istr, v: &istr) {
    let n = istr::to_estr(n);
    let v = istr::to_estr(v);
    let nbuf = str::buf(n);
    let vbuf = str::buf(v);
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
