/*
Module: generic_os

Some miscellaneous platform functions.

These should be rolled into another module.
*/


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
fn getenv(n: str) -> option::t<str> unsafe {
    let s = str::as_buf(n, {|buf| os::libc::getenv(buf) });
    ret if unsafe::reinterpret_cast(s) == 0 {
            option::none::<str>
        } else {
            let s = unsafe::reinterpret_cast(s);
            option::some::<str>(str::str_from_cstr(s))
        };
}

#[cfg(target_os = "linux")]
#[cfg(target_os = "macos")]
fn setenv(n: str, v: str) {
    // FIXME (868)
    let _: () =
        str::as_buf(n,
                    // FIXME (868)
                    {|nbuf|
                        let _: () =
                            str::as_buf(v,
                                        {|vbuf|
                                            os::libc::setenv(nbuf, vbuf, 1i32);
                                        });
                    });
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

// Local Variables:
// mode: rust;
// fill-column: 78;
// indent-tabs-mode: nil
// c-basic-offset: 4
// buffer-file-coding-system: utf-8-unix
// compile-command: "make -k -C .. 2>&1 | sed -e 's/\\/x\\//x:\\//g'";
// End:
