// Some temporary libuv hacks for servo

#[cfg(target_os = "linux")];
#[cfg(target_os = "macos")];
#[cfg(target_os = "freebsd")];


#[nolink]
native mod rustrt {
    fn rust_uvtmp_create_thread() -> thread;
    fn rust_uvtmp_start_thread(thread: thread);
    fn rust_uvtmp_join_thread(thread: thread);
    fn rust_uvtmp_delete_thread(thread: thread);
    fn rust_uvtmp_connect(
        thread: thread,
        ip: str::sbuf,
        chan: comm::chan<iomsg>);
    fn rust_uvtmp_close_connection(thread: thread, cd: connect_data);
    fn rust_uvtmp_write(
        thread: thread,
        cd: connect_data,
        buf: *u8,
        len: ctypes::size_t,
        chan: comm::chan<iomsg>);
    fn rust_uvtmp_read_start(
        thread: thread,
        cd: connect_data,
        chan: comm::chan<iomsg>);
    fn rust_uvtmp_delete_buf(buf: *u8);
}

type thread = *ctypes::void;

type connect_data = *ctypes::void;

enum iomsg {
    whatever,
    connected(connect_data),
    wrote(connect_data),
    read(connect_data, *u8, ctypes::ssize_t)
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

fn connect(thread: thread, ip: str, ch: comm::chan<iomsg>) {
    str::as_buf(ip) {|ipbuf|
        rustrt::rust_uvtmp_connect(thread, ipbuf, ch)
    }
}

fn close_connection(thread: thread, cd: connect_data) {
    rustrt::rust_uvtmp_close_connection(thread ,cd);
}

fn write(thread: thread, cd: connect_data,bytes: [u8],
         chan: comm::chan<iomsg>) unsafe {
    rustrt::rust_uvtmp_write(
        thread, cd, vec::to_ptr(bytes), vec::len(bytes), chan);
}

fn read_start(thread: thread, cd: connect_data,
              chan: comm::chan<iomsg>) {
    rustrt::rust_uvtmp_read_start(thread, cd, chan);
}

fn delete_buf(buf: *u8) {
    rustrt::rust_uvtmp_delete_buf(buf);
}

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
    connect(thread, "74.125.224.146", chan);
    alt comm::recv(port) {
      connected(cd) {
        close_connection(thread, cd);
      }
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
    connect(thread, "74.125.224.146", chan);
    alt comm::recv(port) {
      connected(cd) {
        write(thread, cd, str::bytes("GET / HTTP/1.0\n\n"), chan);
        alt comm::recv(port) {
          wrote(cd) {
            read_start(thread, cd, chan);
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
                        let buf = vec::unsafe::from_buf(buf, len as uint);
                        let str = str::unsafe_from_bytes(buf);
                        #error("read something");
                        io::println(str);
                    }
                    delete_buf(buf);
                  }
                }
            }
            close_connection(thread, cd);
          }
        }
      }
    }
    join_thread(thread);
    delete_thread(thread);
}