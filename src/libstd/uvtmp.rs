// Some temporary libuv hacks for servo

#[nolink]
native mod rustrt {
    fn rust_uvtmp_create_thread() -> thread;
    fn rust_uvtmp_start_thread(thread: thread);
    fn rust_uvtmp_join_thread(thread: thread);
    fn rust_uvtmp_delete_thread(thread: thread);
    fn rust_uvtmp_connect(
        thread: thread,
        req_id: u32,
        ip: str::sbuf,
        chan: comm::chan<iomsg>) -> connect_data;
    fn rust_uvtmp_close_connection(thread: thread, req_id: u32);
    fn rust_uvtmp_write(
        thread: thread,
        req_id: u32,
        buf: *u8,
        len: ctypes::size_t,
        chan: comm::chan<iomsg>);
    fn rust_uvtmp_read_start(
        thread: thread,
        req_id: u32,
        chan: comm::chan<iomsg>);
    fn rust_uvtmp_timer(
        thread: thread,
        timeout: u32,
        req_id: u32,
        chan: comm::chan<iomsg>);
    fn rust_uvtmp_delete_buf(buf: *u8);
    fn rust_uvtmp_get_req_id(cd: connect_data) -> u32;
}

type thread = *ctypes::void;

type connect_data = *ctypes::void;

enum iomsg {
    whatever,
    connected(connect_data),
    wrote(connect_data),
    read(connect_data, *u8, ctypes::ssize_t),
    timer(u32),
    exit
}

fn create_thread() -> thread {
    rustrt::rust_uvtmp_create_thread()
}

fn start_thread(thread: thread) {
    rustrt::rust_uvtmp_start_thread(thread)
}

fn join_thread(thread: thread) {
    rustrt::rust_uvtmp_join_thread(thread)
}

fn delete_thread(thread: thread) {
    rustrt::rust_uvtmp_delete_thread(thread)
}

fn connect(thread: thread, req_id: u32,
           ip: str, ch: comm::chan<iomsg>) -> connect_data {
    str::as_buf(ip) {|ipbuf|
        rustrt::rust_uvtmp_connect(thread, req_id, ipbuf, ch)
    }
}

fn close_connection(thread: thread, req_id: u32) {
    rustrt::rust_uvtmp_close_connection(thread, req_id);
}

fn write(thread: thread, req_id: u32, bytes: [u8],
         chan: comm::chan<iomsg>) unsafe {
    rustrt::rust_uvtmp_write(
        thread, req_id, vec::to_ptr(bytes), vec::len(bytes), chan);
}

fn read_start(thread: thread, req_id: u32,
              chan: comm::chan<iomsg>) {
    rustrt::rust_uvtmp_read_start(thread, req_id, chan);
}

fn timer_start(thread: thread, timeout: u32, req_id: u32,
              chan: comm::chan<iomsg>) {
    rustrt::rust_uvtmp_timer(thread, timeout, req_id, chan);
}

fn delete_buf(buf: *u8) {
    rustrt::rust_uvtmp_delete_buf(buf);
}

fn get_req_id(cd: connect_data) -> u32 {
    ret rustrt::rust_uvtmp_get_req_id(cd);
}

#[cfg(target_os = "linux")]
#[cfg(target_os = "macos")]
#[cfg(target_os = "freebsd")]
// FIXME: We're out of date on libuv and not testing
// it on windows presently. This needs to change.
mod os {

    #[test]
    fn test_start_stop() {
        let thread = create_thread();
        start_thread(thread);
        join_thread(thread);
        delete_thread(thread);
    }

    #[test]
    #[ignore]
    fn test_connect() {
        let thread = create_thread();
        start_thread(thread);
        let port = comm::port();
        let chan = comm::chan(port);
        connect(thread, 0u32, "74.125.224.146", chan);
        alt comm::recv(port) {
          connected(cd) {
            close_connection(thread, 0u32);
          }
          _ { fail "test_connect: port isn't connected"; }
        }
        join_thread(thread);
        delete_thread(thread);
    }

    #[test]
    #[ignore]
    fn test_http() {
        let thread = create_thread();
        start_thread(thread);
        let port = comm::port();
        let chan = comm::chan(port);
        connect(thread, 0u32, "74.125.224.146", chan);
        alt comm::recv(port) {
          connected(cd) {
            write(thread, 0u32, str::bytes("GET / HTTP/1.0\n\n"), chan);
            alt comm::recv(port) {
              wrote(cd) {
                read_start(thread, 0u32, chan);
                let keep_going = true;
                while keep_going {
                    alt comm::recv(port) {
                      read(_, buf, -1) {
                        keep_going = false;
                        delete_buf(buf);
                      }
                      read(_, buf, len) {
                        unsafe {
                            log(error, len);
                            let buf = vec::unsafe::from_buf(buf,
                                                            len as uint);
                            let str = str::from_bytes(buf);
                            #error("read something");
                            io::println(str);
                        }
                        delete_buf(buf);
                      }
                      _ { fail "test_http: protocol error"; }
                    }
                }
                close_connection(thread, 0u32);
              }
              _ { fail "test_http: expected `wrote`"; }
            }
          }
          _ { fail "test_http: port not connected"; }
        }
        join_thread(thread);
        delete_thread(thread);
    }
}