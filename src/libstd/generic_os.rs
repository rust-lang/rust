/*
Module: generic_os

Some miscellaneous platform functions.

These should be rolled into another module.
*/

import core::option;

// Wow, this is an ugly way to write doc comments

#[cfg(bogus)]
#[doc = "Get the value of an environment variable"]
fn getenv(n: str) -> option<str> { }

#[cfg(bogus)]
#[doc = "Set the value of an environment variable"]
fn setenv(n: str, v: str) { }

fn env() -> [(str,str)] {
    let pairs = [];
    for p in os::rustrt::rust_env_pairs() {
        let vs = str::splitn_char(p, '=', 1u);
        assert vec::len(vs) == 2u;
        pairs += [(vs[0], vs[1])];
    }
    ret pairs;
}

#[cfg(target_os = "linux")]
#[cfg(target_os = "macos")]
#[cfg(target_os = "freebsd")]
fn getenv(n: str) -> option<str> unsafe {
    let s = str::as_buf(n, {|buf| os::libc::getenv(buf) });
    ret if unsafe::reinterpret_cast(s) == 0 {
            option::none::<str>
        } else {
            let s = unsafe::reinterpret_cast(s);
            option::some::<str>(str::from_cstr(s))
        };
}

#[cfg(target_os = "linux")]
#[cfg(target_os = "macos")]
#[cfg(target_os = "freebsd")]
fn setenv(n: str, v: str) {
    // FIXME (868)
    str::as_buf(
        n,
        // FIXME (868)
        {|nbuf|
            str::as_buf(
                v,
                {|vbuf|
                    os::libc::setenv(nbuf, vbuf, 1i32)})});
}

#[cfg(target_os = "win32")]
fn getenv(n: str) -> option<str> {
    let nsize = 256u;
    while true {
        let v: [u8] = [];
        vec::reserve(v, nsize);
        let res =
            str::as_buf(n,
                        {|nbuf|
                            unsafe {
                            let vbuf = vec::unsafe::to_ptr(v);
                            os::kernel32::GetEnvironmentVariableA(nbuf, vbuf,
                                                                  nsize)
                        }
                        });
        if res == 0u {
            ret option::none;
        } else if res < nsize {
            unsafe {
                vec::unsafe::set_len(v, res);
            }
            ret option::some(str::from_bytes(v)); // UTF-8 or fail
        } else { nsize = res; }
    }
    util::unreachable();
}

#[cfg(target_os = "win32")]
fn setenv(n: str, v: str) {
    // FIXME (868)
    let _: () =
        str::as_buf(n, {|nbuf|
            let _: () =
                str::as_buf(v, {|vbuf|
                    os::kernel32::SetEnvironmentVariableA(nbuf, vbuf);
                });
        });
}


#[cfg(test)]
mod tests {

    fn make_rand_name() -> str {
        import rand;
        let rng: rand::rng = rand::mk_rng();
        let n = "TEST" + rng.gen_str(10u);
        assert option::is_none(getenv(n));
        n
    }

    #[test]
    #[ignore(reason = "fails periodically on mac")]
    fn test_setenv() {
        let n = make_rand_name();
        setenv(n, "VALUE");
        assert getenv(n) == option::some("VALUE");
    }

    #[test]
    #[ignore(reason = "fails periodically on mac")]
    fn test_setenv_overwrite() {
        let n = make_rand_name();
        setenv(n, "1");
        setenv(n, "2");
        assert getenv(n) == option::some("2");
        setenv(n, "");
        assert getenv(n) == option::some("");
    }

    // Windows GetEnvironmentVariable requires some extra work to make sure
    // the buffer the variable is copied into is the right size
    #[test]
    #[ignore(reason = "fails periodically on mac")]
    fn test_getenv_big() {
        let s = "";
        let i = 0;
        while i < 100 { s += "aaaaaaaaaa"; i += 1; }
        let n = make_rand_name();
        setenv(n, s);
        log(debug, s);
        assert getenv(n) == option::some(s);
    }

    #[test]
    fn test_get_exe_path() {
        let path = os::get_exe_path();
        assert option::is_some(path);
        let path = option::get(path);
        log(debug, path);

        // Hard to test this function
        if os::target_os() != "win32" {
            assert str::starts_with(path, fs::path_sep());
        } else {
            assert path[1] == ':' as u8;
        }
    }

    #[test]
    fn test_env_getenv() {
        let e = env();
        assert vec::len(e) > 0u;
        for (n, v) in e {
            log(debug, n);
            let v2 = getenv(n);
            // MingW seems to set some funky environment variables like
            // "=C:=C:\MinGW\msys\1.0\bin" and "!::=::\" that are returned
            // from env() but not visible from getenv().
            assert option::is_none(v2) || v2 == option::some(v);
        }
    }

    #[test]
    fn test_env_setenv() {
        let n = make_rand_name();

        let e = env();
        setenv(n, "VALUE");
        assert !vec::contains(e, (n, "VALUE"));

        e = env();
        assert vec::contains(e, (n, "VALUE"));
    }
}

// Local Variables:
// mode: rust;
// fill-column: 78;
// indent-tabs-mode: nil
// c-basic-offset: 4
// buffer-file-coding-system: utf-8-unix
// End:
