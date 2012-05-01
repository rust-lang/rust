#[doc = "Internal module for serializing access to getenv/setenv"];

export getenv;
export setenv;

native mod rustrt {
    fn rust_global_env_chan_ptr() -> *libc::uintptr_t;
}

enum msg {
    msg_getenv(str, comm::chan<option<str>>),
    msg_setenv(str, str, comm::chan<()>)
}

fn getenv(n: str) -> option<str> {
    let env_ch = get_global_env_chan();
    let po = comm::port();
    comm::send(env_ch, msg_getenv(n, comm::chan(po)));
    comm::recv(po)
}

fn setenv(n: str, v: str) {
    let env_ch = get_global_env_chan();
    let po = comm::port();
    comm::send(env_ch, msg_setenv(n, v, comm::chan(po)));
    comm::recv(po)
}

fn get_global_env_chan() -> comm::chan<msg> {
    let global_ptr = rustrt::rust_global_env_chan_ptr();
    let builder_fn = {||
        let builder = task::builder();
        task::unsupervise(builder);
        task::set_opts(builder, {
            sched:  some({
                mode: task::single_threaded,
                // FIXME: This would be a good place to use
                // a very small native stack
                native_stack_size: none
            })
            with task::get_opts(builder)
        });
        builder
    };
    unsafe {
        priv::chan_from_global_ptr(
            global_ptr, builder_fn, global_env_task)
    }
}

fn global_env_task(msg_po: comm::port<msg>) unsafe {
    priv::weaken_task {|weak_po|
        loop {
            alt comm::select2(msg_po, weak_po) {
              either::left(msg_getenv(n, resp_ch)) {
                comm::send(resp_ch, impl::getenv(n))
              }
              either::left(msg_setenv(n, v, resp_ch)) {
                comm::send(resp_ch, impl::setenv(n, v))
              }
              either::right(_) {
                break;
              }
            }
        }
    }
}

mod impl {

    #[cfg(target_os = "linux")]
    #[cfg(target_os = "macos")]
    #[cfg(target_os = "freebsd")]
    fn getenv(n: str) -> option<str> unsafe {
        let s = str::as_c_str(n, libc::getenv);
        ret if unsafe::reinterpret_cast(s) == 0 {
            option::none::<str>
        } else {
            let s = unsafe::reinterpret_cast(s);
            option::some::<str>(str::unsafe::from_buf(s))
        };
    }

    #[cfg(target_os = "win32")]
    fn getenv(n: str) -> option<str> unsafe {
        import libc::types::os::arch::extra::*;
        import libc::funcs::extra::kernel32::*;
        import win32::*;
        as_utf16_p(n) {|u|
            fill_utf16_buf_and_decode() {|buf, sz|
                GetEnvironmentVariableW(u, buf, sz)
            }
        }
    }


    #[cfg(target_os = "linux")]
    #[cfg(target_os = "macos")]
    #[cfg(target_os = "freebsd")]
    fn setenv(n: str, v: str) {

        // FIXME: remove this when export globs work properly.
        import libc::funcs::posix01::unistd::setenv;
        str::as_c_str(n) {|nbuf|
            str::as_c_str(v) {|vbuf|
                setenv(nbuf, vbuf, 1i32);
            }
        }
    }


    #[cfg(target_os = "win32")]
    fn setenv(n: str, v: str) {
        // FIXME: remove imports when export globs work properly.
        import libc::funcs::extra::kernel32::*;
        import win32::*;
        as_utf16_p(n) {|nbuf|
            as_utf16_p(v) {|vbuf|
                SetEnvironmentVariableW(nbuf, vbuf);
            }
        }
    }

}