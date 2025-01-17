use std::time::Instant;

#[cfg(not(target_arch = "wasm32"))]
use test::{Bencher, black_box};

macro_rules! bench_instant_threaded {
    ($bench_name:ident, $thread_count:expr) => {
        #[bench]
        #[cfg(not(target_arch = "wasm32"))]
        fn $bench_name(b: &mut Bencher) -> std::thread::Result<()> {
            use std::sync::Arc;
            use std::sync::atomic::{AtomicBool, Ordering};

            let running = Arc::new(AtomicBool::new(true));

            let threads: Vec<_> = (0..$thread_count)
                .map(|_| {
                    let flag = Arc::clone(&running);
                    std::thread::spawn(move || {
                        while flag.load(Ordering::Relaxed) {
                            black_box(Instant::now());
                        }
                    })
                })
                .collect();

            b.iter(|| {
                let a = Instant::now();
                let b = Instant::now();
                assert!(b >= a);
            });

            running.store(false, Ordering::Relaxed);

            for t in threads {
                t.join()?;
            }
            Ok(())
        }
    };
}

bench_instant_threaded!(instant_contention_01_threads, 0);
bench_instant_threaded!(instant_contention_02_threads, 1);
bench_instant_threaded!(instant_contention_04_threads, 3);
bench_instant_threaded!(instant_contention_08_threads, 7);
bench_instant_threaded!(instant_contention_16_threads, 15);
