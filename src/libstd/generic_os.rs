/*
Module: generic_os

Some miscellaneous platform functions.

These should be rolled into another module.
*/

import core::option;

// Wow, this is an ugly way to write doc comments

#[cfg(bogus)]
/*
Function: getenv

Get the value of an environment variable
*/
fn getenv(n: str) -> option::t<str> { }

#[cfg(bogus)]
/*
Function: setenv

Set the value of an environment variable
*/
fn setenv(n: str, v: str) { }

#[cfg(target_os = "linux")]
#[cfg(target_os = "macos")]
#[cfg(target_os = "freebsd")]
fn getenv(n: str) -> option::t<str> unsafe {
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
fn getenv(n: str) -> option::t<str> {
    let nsize = 256u;
    while true {
        let v: [u8] = [];
        vec::reserve(v, nsize);
        let res =
            str::as_buf(n,
                        {|nbuf|
                            unsafe {
                            let vbuf = vec::to_ptr(v);
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
            ret option::some(str::unsafe_from_bytes(v));
        } else { nsize = res; }
    }
    fail;
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

    #[test]
    #[ignore(reason = "fails periodically on mac")]
    fn test_setenv() {
        // NB: Each test of setenv needs to use different variable names or
        // the tests will not be threadsafe
        setenv("NAME1", "VALUE");
        assert (getenv("NAME1") == option::some("VALUE"));
    }

    #[test]
    #[ignore(reason = "fails periodically on mac")]
    fn test_setenv_overwrite() {
        setenv("NAME2", "1");
        setenv("NAME2", "2");
        assert (getenv("NAME2") == option::some("2"));
    }

    // Windows GetEnvironmentVariable requires some extra work to make sure
    // the buffer the variable is copied into is the right size
    #[test]
    #[ignore(reason = "fails periodically on mac")]
    fn test_getenv_big() {
        let s = "";
        let i = 0;
        while i < 100 { s += "aaaaaaaaaa"; i += 1; }
        setenv("test_getenv_big", s);
        log(debug, s);
        assert (getenv("test_getenv_big") == option::some(s));
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
}

// Local Variables:
// mode: rust;
// fill-column: 78;
// indent-tabs-mode: nil
// c-basic-offset: 4
// buffer-file-coding-system: utf-8-unix
// End:
