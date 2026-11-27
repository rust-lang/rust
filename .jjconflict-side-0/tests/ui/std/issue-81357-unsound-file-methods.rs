//@ run-crash
//@ only-windows

fn main() {
    use std::fs;
    use std::io::prelude::*;
    use std::os::windows::prelude::*;
    use std::ptr;
    use std::sync::Arc;
    use std::thread;
    use std::time::Duration;

    const FILE_FLAG_OVERLAPPED: u32 = 0x40000000;

    fn create_pipe_server(path: &str) -> fs::File {
        let mut path0 = path.as_bytes().to_owned();
        path0.push(0);
        extern "system" {
            fn CreateNamedPipeA(
                lpName: *const u8,
                dwOpenMode: u32,
                dwPipeMode: u32,
                nMaxInstances: u32,
                nOutBufferSize: u32,
                nInBufferSize: u32,
                nDefaultTimeOut: u32,
                lpSecurityAttributes: *mut u8,
            ) -> RawHandle;
        }

        unsafe {
            let h = CreateNamedPipeA(path0.as_ptr(), 3, 0, 1, 0, 0, 0, ptr::null_mut());
            assert_ne!(h as isize, -1);
            fs::File::from_raw_handle(h)
        }
    }

    let path = "\\\\.\\pipe\\repro";
    let mut server = create_pipe_server(path);

    let client = Arc::new(
        fs::OpenOptions::new().custom_flags(FILE_FLAG_OVERLAPPED).read(true).open(path).unwrap(),
    );

    let spawn_read = |is_first: bool| {
        thread::spawn({
            let f = client.clone();
            move || {
                let mut buf = [0xcc; 1];
                let mut f = f.as_ref();
                f.read(&mut buf).unwrap();
                if is_first {
                    assert_ne!(buf[0], 0xcc);
                } else {
                    let b = buf[0]; // capture buf[0]
                    thread::sleep(Duration::from_millis(200));

                    // Check the buffer hasn't been written to after read.
                    dbg!(buf[0], b);
                    assert_eq!(buf[0], b);
                }
            }
        })
    };

    let t1 = spawn_read(true);
    thread::sleep(Duration::from_millis(20));
    let t2 = spawn_read(false);
    thread::sleep(Duration::from_millis(100));
    let _ = server.write(b"x");
    thread::sleep(Duration::from_millis(100));
    let _ = server.write(b"y");

    // This is run fail because we need to test for the `abort`.
    // That failing to run is the success case.
    if t1.join().is_err() || t2.join().is_err() {
        return;
    } else {
        panic!("success");
    }
}
