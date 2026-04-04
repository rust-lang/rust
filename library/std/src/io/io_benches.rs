use crate::io::prelude::*;
use crate::io::{self, BorrowedBuf, IoSlice};
use crate::mem::MaybeUninit;

#[bench]
fn bench_read_slice(b: &mut test::Bencher) {
    let buf = [5; 1024];
    let mut dst = [0; 128];

    b.iter(|| {
        let mut rd = &buf[..];
        for _ in 0..8 {
            let _ = rd.read(&mut dst);
            test::black_box(&dst);
        }
    })
}

#[bench]
fn bench_write_slice(b: &mut test::Bencher) {
    let mut buf = [0; 1024];
    let src = [5; 128];

    b.iter(|| {
        let mut wr = &mut buf[..];
        for _ in 0..8 {
            let _ = wr.write_all(&src);
            test::black_box(&wr);
        }
    })
}

#[bench]
fn bench_read_vec(b: &mut test::Bencher) {
    let buf = vec![5; 1024];
    let mut dst = [0; 128];

    b.iter(|| {
        let mut rd = &buf[..];
        for _ in 0..8 {
            let _ = rd.read(&mut dst);
            test::black_box(&dst);
        }
    })
}

#[bench]
fn bench_write_vec(b: &mut test::Bencher) {
    let mut buf = Vec::with_capacity(1024);
    let src = [5; 128];

    b.iter(|| {
        let mut wr = &mut buf[..];
        for _ in 0..8 {
            let _ = wr.write_all(&src);
            test::black_box(&wr);
        }
    })
}

#[cfg(unix)]
#[cfg_attr(target_os = "emscripten", ignore)] // no /dev
#[bench]
fn bench_copy_buf_reader(b: &mut test::Bencher) {
    use crate::fs;

    let mut file_in = fs::File::open("/dev/zero").expect("opening /dev/zero failed");
    // use dyn to avoid specializations unrelated to readbuf
    let dyn_in = &mut file_in as &mut dyn Read;
    let mut reader = io::BufReader::with_capacity(256 * 1024, dyn_in.take(0));
    let mut writer =
        fs::OpenOptions::new().write(true).open("/dev/null").expect("opening /dev/null failed");

    const BYTES: u64 = 1024 * 1024;

    b.bytes = BYTES;

    b.iter(|| {
        reader.get_mut().set_limit(BYTES);
        io::copy(&mut reader, &mut writer).unwrap()
    });
}

#[bench]
fn bench_write_cursor_vec(b: &mut test::Bencher) {
    let slice = &[1; 128];

    b.iter(|| {
        let mut buf = b"some random data to overwrite".to_vec();
        let mut cursor = io::Cursor::new(&mut buf);

        let _ = cursor.write_all(slice);
        test::black_box(&cursor);
    })
}

#[bench]
fn bench_write_cursor_vec_vectored(b: &mut test::Bencher) {
    let slices = [
        IoSlice::new(&[1; 128]),
        IoSlice::new(&[2; 256]),
        IoSlice::new(&[3; 512]),
        IoSlice::new(&[4; 1024]),
        IoSlice::new(&[5; 2048]),
        IoSlice::new(&[6; 4096]),
        IoSlice::new(&[7; 8192]),
        IoSlice::new(&[8; 8192 * 2]),
    ];

    b.iter(|| {
        let mut buf = b"some random data to overwrite".to_vec();
        let mut cursor = io::Cursor::new(&mut buf);

        let mut slices = slices;
        let _ = cursor.write_all_vectored(&mut slices);
        test::black_box(&cursor);
    })
}

#[bench]
fn bench_take_read(b: &mut test::Bencher) {
    b.iter(|| {
        let mut buf = [0; 64];

        [255; 128].take(64).read(&mut buf).unwrap();
    });
}

#[bench]
fn bench_take_read_buf(b: &mut test::Bencher) {
    b.iter(|| {
        let buf: &mut [_] = &mut [MaybeUninit::uninit(); 64];

        let mut buf: BorrowedBuf<'_> = buf.into();

        [255; 128].take(64).read_buf(buf.unfilled()).unwrap();
    });
}

#[bench]
#[cfg_attr(miri, ignore)] // Miri isn't fast...
fn bench_read_to_end(b: &mut test::Bencher) {
    b.iter(|| {
        let mut lr = io::repeat(1).take(10000000);
        let mut vec = Vec::with_capacity(1024);
        io::default_read_to_end(&mut lr, &mut vec, None)
    });
}

#[bench]
fn bench_buffered_reader(b: &mut test::Bencher) {
    b.iter(|| io::BufReader::new(io::empty()));
}

#[bench]
fn bench_buffered_reader_small_reads(b: &mut test::Bencher) {
    let data = (0..u8::MAX).cycle().take(1024 * 4).collect::<Vec<_>>();
    b.iter(|| {
        let mut reader = io::BufReader::new(&data[..]);
        let mut buf = [0u8; 4];
        for _ in 0..1024 {
            reader.read_exact(&mut buf).unwrap();
            core::hint::black_box(&buf);
        }
    });
}

#[bench]
fn bench_buffered_writer(b: &mut test::Bencher) {
    b.iter(|| io::BufWriter::new(io::sink()));
}
