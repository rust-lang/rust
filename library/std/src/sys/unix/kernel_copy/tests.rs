use crate::env::temp_dir;
use crate::fs::OpenOptions;
use crate::io;
use crate::io::Result;
use crate::io::SeekFrom;
use crate::io::{BufRead, Read, Seek, Write};
use crate::os::unix::io::AsRawFd;

#[test]
fn copy_specialization() -> Result<()> {
    use crate::io::{BufReader, BufWriter};

    let path = crate::env::temp_dir();
    let source_path = path.join("copy-spec.source");
    let sink_path = path.join("copy-spec.sink");

    let result: Result<()> = try {
        let mut source = crate::fs::OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .truncate(true)
            .open(&source_path)?;
        source.write_all(b"abcdefghiklmnopqr")?;
        source.seek(SeekFrom::Start(8))?;
        let mut source = BufReader::with_capacity(8, source.take(5));
        source.fill_buf()?;
        assert_eq!(source.buffer(), b"iklmn");
        source.get_mut().set_limit(6);
        source.get_mut().get_mut().seek(SeekFrom::Start(1))?; // "bcdefg"
        let mut source = source.take(10); // "iklmnbcdef"

        let mut sink = crate::fs::OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .truncate(true)
            .open(&sink_path)?;
        sink.write_all(b"000000")?;
        let mut sink = BufWriter::with_capacity(5, sink);
        sink.write_all(b"wxyz")?;
        assert_eq!(sink.buffer(), b"wxyz");

        let copied = crate::io::copy(&mut source, &mut sink)?;
        assert_eq!(copied, 10);
        assert_eq!(sink.buffer().len(), 0);

        let mut sink = sink.into_inner()?;
        sink.seek(SeekFrom::Start(0))?;
        let mut copied = Vec::new();
        sink.read_to_end(&mut copied)?;
        assert_eq!(&copied, b"000000wxyziklmnbcdef");
    };

    let rm1 = crate::fs::remove_file(source_path);
    let rm2 = crate::fs::remove_file(sink_path);

    result.and(rm1).and(rm2)
}

#[bench]
fn bench_file_to_file_copy(b: &mut test::Bencher) {
    const BYTES: usize = 128 * 1024;
    let src_path = temp_dir().join("file-copy-bench-src");
    let mut src = crate::fs::OpenOptions::new()
        .create(true)
        .truncate(true)
        .read(true)
        .write(true)
        .open(src_path)
        .unwrap();
    src.write(&vec![0u8; BYTES]).unwrap();

    let sink_path = temp_dir().join("file-copy-bench-sink");
    let mut sink = crate::fs::OpenOptions::new()
        .create(true)
        .truncate(true)
        .write(true)
        .open(sink_path)
        .unwrap();

    b.bytes = BYTES as u64;
    b.iter(|| {
        src.seek(SeekFrom::Start(0)).unwrap();
        sink.seek(SeekFrom::Start(0)).unwrap();
        assert_eq!(BYTES as u64, io::copy(&mut src, &mut sink).unwrap());
    });
}

#[bench]
fn bench_file_to_socket_copy(b: &mut test::Bencher) {
    const BYTES: usize = 128 * 1024;
    let src_path = temp_dir().join("pipe-copy-bench-src");
    let mut src = OpenOptions::new()
        .create(true)
        .truncate(true)
        .read(true)
        .write(true)
        .open(src_path)
        .unwrap();
    src.write(&vec![0u8; BYTES]).unwrap();

    let sink_drainer = crate::net::TcpListener::bind("localhost:0").unwrap();
    let mut sink = crate::net::TcpStream::connect(sink_drainer.local_addr().unwrap()).unwrap();
    let mut sink_drainer = sink_drainer.accept().unwrap().0;

    crate::thread::spawn(move || {
        let mut sink_buf = vec![0u8; 1024 * 1024];
        loop {
            sink_drainer.read(&mut sink_buf[..]).unwrap();
        }
    });

    b.bytes = BYTES as u64;
    b.iter(|| {
        src.seek(SeekFrom::Start(0)).unwrap();
        assert_eq!(BYTES as u64, io::copy(&mut src, &mut sink).unwrap());
    });
}

#[cfg(any(target_os = "linux", target_os = "android"))]
#[bench]
fn bench_socket_pipe_socket_copy(b: &mut test::Bencher) {
    use super::CopyResult;
    use crate::io::ErrorKind;
    use crate::process::{ChildStdin, ChildStdout};
    use crate::sys_common::FromInner;

    let (read_end, write_end) = crate::sys::pipe::anon_pipe().unwrap();

    let mut read_end = ChildStdout::from_inner(read_end);
    let write_end = ChildStdin::from_inner(write_end);

    let acceptor = crate::net::TcpListener::bind("localhost:0").unwrap();
    let mut remote_end = crate::net::TcpStream::connect(acceptor.local_addr().unwrap()).unwrap();

    let local_end = crate::sync::Arc::new(acceptor.accept().unwrap().0);

    // the data flow in this benchmark:
    //
    //                      socket(tx)  local_source
    // remote_end (write)  +-------->   (splice to)
    //                                  write_end
    //                                     +
    //                                     |
    //                                     | pipe
    //                                     v
    //                                  read_end
    // remote_end (read)   <---------+  (splice to) *
    //                      socket(rx)  local_end
    //
    // * benchmark loop using io::copy

    crate::thread::spawn(move || {
        let mut sink_buf = vec![0u8; 1024 * 1024];
        remote_end.set_nonblocking(true).unwrap();
        loop {
            match remote_end.write(&mut sink_buf[..]) {
                Err(err) if err.kind() == ErrorKind::WouldBlock => {}
                Ok(_) => {}
                err => {
                    err.expect("write failed");
                }
            };
            match remote_end.read(&mut sink_buf[..]) {
                Err(err) if err.kind() == ErrorKind::WouldBlock => {}
                Ok(_) => {}
                err => {
                    err.expect("read failed");
                }
            };
        }
    });

    // check that splice works, otherwise the benchmark would hang
    let probe = super::sendfile_splice(
        super::SpliceMode::Splice,
        local_end.as_raw_fd(),
        write_end.as_raw_fd(),
        1,
    );

    match probe {
        CopyResult::Ended(Ok(1)) => {
            // splice works
        }
        _ => {
            eprintln!("splice failed, skipping benchmark");
            return;
        }
    }

    let local_source = local_end.clone();
    crate::thread::spawn(move || {
        loop {
            super::sendfile_splice(
                super::SpliceMode::Splice,
                local_source.as_raw_fd(),
                write_end.as_raw_fd(),
                u64::MAX,
            );
        }
    });

    const BYTES: usize = 128 * 1024;
    b.bytes = BYTES as u64;
    b.iter(|| {
        assert_eq!(
            BYTES as u64,
            io::copy(&mut (&mut read_end).take(BYTES as u64), &mut &*local_end).unwrap()
        );
    });
}
