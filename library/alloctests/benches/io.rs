use std::io::prelude::*;

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

#[bench]
#[cfg(unix)]
#[cfg_attr(target_os = "emscripten", ignore)] // no /dev
fn bench_copy_buf_reader(b: &mut test::Bencher) {
    use std::fs::{File, OpenOptions};

    let mut file_in = File::open("/dev/zero").expect("opening /dev/zero failed");
    // use dyn to avoid specializations unrelated to readbuf
    let dyn_in = &mut file_in as &mut dyn Read;
    let mut reader = std::io::BufReader::with_capacity(256 * 1024, dyn_in.take(0));
    let mut writer =
        OpenOptions::new().write(true).open("/dev/null").expect("opening /dev/null failed");

    const BYTES: u64 = 1024 * 1024;

    b.bytes = BYTES;

    b.iter(|| {
        reader.get_mut().set_limit(BYTES);
        std::io::copy(&mut reader, &mut writer).unwrap()
    });
}
