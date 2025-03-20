//! Benchmarks that use Callgrind (via `iai_callgrind`) to report instruction count metrics. This
//! is stable enough to be tested in CI.

use std::hint::black_box;
use std::{ops, slice};

use compiler_builtins::mem::{memcmp, memcpy, memmove, memset};
use iai_callgrind::{library_benchmark, library_benchmark_group, main};

const PAGE_SIZE: usize = 0x1000;

#[derive(Clone)]
#[repr(C, align(0x1000))]
struct Page([u8; PAGE_SIZE]);

/// A buffer that is page-aligned by default, with an optional offset to create a
/// misalignment.
struct AlignedSlice {
    buf: Box<[Page]>,
    len: usize,
    offset: usize,
}

impl AlignedSlice {
    /// Allocate a slice aligned to ALIGN with at least `len` items, with `offset` from
    /// page alignment.
    fn new_zeroed(len: usize, offset: usize) -> Self {
        assert!(offset < PAGE_SIZE);
        let total_len = len + offset;
        let items = (total_len / PAGE_SIZE) + if total_len % PAGE_SIZE > 0 { 1 } else { 0 };
        let buf = vec![Page([0u8; PAGE_SIZE]); items].into_boxed_slice();
        AlignedSlice { buf, len, offset }
    }
}

impl ops::Deref for AlignedSlice {
    type Target = [u8];
    fn deref(&self) -> &Self::Target {
        unsafe { slice::from_raw_parts(self.buf.as_ptr().cast::<u8>().add(self.offset), self.len) }
    }
}

impl ops::DerefMut for AlignedSlice {
    fn deref_mut(&mut self) -> &mut Self::Target {
        unsafe {
            slice::from_raw_parts_mut(
                self.buf.as_mut_ptr().cast::<u8>().add(self.offset),
                self.len,
            )
        }
    }
}

mod mcpy {
    use super::*;

    struct Cfg {
        len: usize,
        s_off: usize,
        d_off: usize,
    }

    fn setup(cfg: Cfg) -> (usize, AlignedSlice, AlignedSlice) {
        let Cfg { len, s_off, d_off } = cfg;
        println!("{len} bytes, {s_off} src offset, {d_off} dst offset");
        let mut src = AlignedSlice::new_zeroed(len, s_off);
        let dst = AlignedSlice::new_zeroed(len, d_off);
        src.fill(1);
        (len, src, dst)
    }

    #[library_benchmark]
    #[benches::aligned(
        args = [
            Cfg { len: 16, s_off: 0, d_off: 0 },
            Cfg { len: 16, s_off: 0, d_off: 0 },
            Cfg { len: 28, s_off: 0, d_off: 0 },
            Cfg { len: 32, s_off: 0, d_off: 0 },
            Cfg { len: 36, s_off: 0, d_off: 0 },
            Cfg { len: 60, s_off: 0, d_off: 0 },
            Cfg { len: 64, s_off: 0, d_off: 0 },
            Cfg { len: 68, s_off: 0, d_off: 0 },
            Cfg { len: 128, s_off: 0, d_off: 0 },
            Cfg { len: 256, s_off: 0, d_off: 0 },
            Cfg { len: 512, s_off: 0, d_off: 0 },
            Cfg { len: 1024, s_off: 0, d_off: 0 },
            Cfg { len: 4096, s_off: 0, d_off: 0 },
            Cfg { len: 1048576, s_off: 0, d_off: 0 },
        ],
        setup = setup,
    )]
    #[benches::offset(
        args = [
            Cfg { len: 16, s_off: 65, d_off: 65 },
            Cfg { len: 28, s_off: 65, d_off: 65 },
            Cfg { len: 32, s_off: 65, d_off: 65 },
            Cfg { len: 36, s_off: 65, d_off: 65 },
            Cfg { len: 60, s_off: 65, d_off: 65 },
            Cfg { len: 64, s_off: 65, d_off: 65 },
            Cfg { len: 68, s_off: 65, d_off: 65 },
            Cfg { len: 128, s_off: 65, d_off: 65 },
            Cfg { len: 256, s_off: 65, d_off: 65 },
            Cfg { len: 512, s_off: 65, d_off: 65 },
            Cfg { len: 1024, s_off: 65, d_off: 65 },
            Cfg { len: 4096, s_off: 65, d_off: 65 },
            Cfg { len: 1048576, s_off: 65, d_off: 65 },
        ],
        setup = setup,
    )]
    #[benches::misaligned(
        args = [
            Cfg { len: 16, s_off: 65, d_off: 66 },
            Cfg { len: 28, s_off: 65, d_off: 66 },
            Cfg { len: 32, s_off: 65, d_off: 66 },
            Cfg { len: 36, s_off: 65, d_off: 66 },
            Cfg { len: 60, s_off: 65, d_off: 66 },
            Cfg { len: 64, s_off: 65, d_off: 66 },
            Cfg { len: 68, s_off: 65, d_off: 66 },
            Cfg { len: 128, s_off: 65, d_off: 66 },
            Cfg { len: 256, s_off: 65, d_off: 66 },
            Cfg { len: 512, s_off: 65, d_off: 66 },
            Cfg { len: 1024, s_off: 65, d_off: 66 },
            Cfg { len: 4096, s_off: 65, d_off: 66 },
            Cfg { len: 1048576, s_off: 65, d_off: 66 },
        ],
        setup = setup,
    )]
    fn bench((len, mut dst, src): (usize, AlignedSlice, AlignedSlice)) {
        unsafe {
            black_box(memcpy(
                black_box(dst.as_mut_ptr()),
                black_box(src.as_ptr()),
                black_box(len),
            ));
        }
    }

    library_benchmark_group!(name = memcpy; benchmarks = bench);
}

mod mset {
    use super::*;

    struct Cfg {
        len: usize,
        offset: usize,
    }

    fn setup(Cfg { len, offset }: Cfg) -> (usize, AlignedSlice) {
        println!("{len} bytes, {offset} offset");
        (len, AlignedSlice::new_zeroed(len, offset))
    }

    #[library_benchmark]
    #[benches::aligned(
        args = [
            Cfg { len: 16, offset: 0 },
            Cfg { len: 32, offset: 0 },
            Cfg { len: 64, offset: 0 },
            Cfg { len: 512, offset: 0 },
            Cfg { len: 4096, offset: 0 },
            Cfg { len: 1048576, offset: 0 },
        ],
        setup = setup,
    )]
    #[benches::offset(
        args = [
            Cfg { len: 16, offset: 65 },
            Cfg { len: 32, offset: 65 },
            Cfg { len: 64, offset: 65 },
            Cfg { len: 512, offset: 65 },
            Cfg { len: 4096, offset: 65 },
            Cfg { len: 1048576, offset: 65 },
        ],
        setup = setup,
    )]
    fn bench((len, mut dst): (usize, AlignedSlice)) {
        unsafe {
            black_box(memset(
                black_box(dst.as_mut_ptr()),
                black_box(27),
                black_box(len),
            ));
        }
    }

    library_benchmark_group!(name = memset; benchmarks = bench);
}

mod mcmp {
    use super::*;

    struct Cfg {
        len: usize,
        s_off: usize,
        d_off: usize,
    }

    fn setup(cfg: Cfg) -> (usize, AlignedSlice, AlignedSlice) {
        let Cfg { len, s_off, d_off } = cfg;
        println!("{len} bytes, {s_off} src offset, {d_off} dst offset");
        let b1 = AlignedSlice::new_zeroed(len, s_off);
        let mut b2 = AlignedSlice::new_zeroed(len, d_off);
        b2[len - 1] = 1;
        (len, b1, b2)
    }

    #[library_benchmark]
    #[benches::aligned(
        args = [
            Cfg { len: 16, s_off: 0, d_off: 0 },
            Cfg { len: 32, s_off: 0, d_off: 0 },
            Cfg { len: 64, s_off: 0, d_off: 0 },
            Cfg { len: 512, s_off: 0, d_off: 0 },
            Cfg { len: 4096, s_off: 0, d_off: 0 },
            Cfg { len: 1048576, s_off: 0, d_off: 0 },
        ],
        setup = setup
    )]
    #[benches::offset(
        args = [
            Cfg { len: 16, s_off: 65, d_off: 65 },
            Cfg { len: 32, s_off: 65, d_off: 65 },
            Cfg { len: 64, s_off: 65, d_off: 65 },
            Cfg { len: 512, s_off: 65, d_off: 65 },
            Cfg { len: 4096, s_off: 65, d_off: 65 },
            Cfg { len: 1048576, s_off: 65, d_off: 65 },
        ],
        setup = setup
    )]
    #[benches::misaligned(
        args = [
            Cfg { len: 16, s_off: 65, d_off: 66 },
            Cfg { len: 32, s_off: 65, d_off: 66 },
            Cfg { len: 64, s_off: 65, d_off: 66 },
            Cfg { len: 512, s_off: 65, d_off: 66 },
            Cfg { len: 4096, s_off: 65, d_off: 66 },
            Cfg { len: 1048576, s_off: 65, d_off: 66 },
        ],
        setup = setup
    )]
    fn bench((len, mut dst, src): (usize, AlignedSlice, AlignedSlice)) {
        unsafe {
            black_box(memcmp(
                black_box(dst.as_mut_ptr()),
                black_box(src.as_ptr()),
                black_box(len),
            ));
        }
    }

    library_benchmark_group!(name = memcmp; benchmarks = bench);
}

mod mmove {
    use super::*;
    use Spread::{Large, Medium, Small};

    struct Cfg {
        len: usize,
        spread: Spread,
        off: usize,
    }

    enum Spread {
        /// `src` and `dst` are close.
        Small,
        /// `src` and `dst` are halfway offset in the buffer.
        Medium,
        /// `src` and `dst` only overlap by a single byte.
        Large,
    }

    fn calculate_spread(len: usize, spread: Spread) -> usize {
        match spread {
            Small => 1,
            Medium => len / 2,
            Large => len - 1,
        }
    }

    fn setup_forward(cfg: Cfg) -> (usize, usize, AlignedSlice) {
        let Cfg { len, spread, off } = cfg;
        let spread = calculate_spread(len, spread);
        println!("{len} bytes, {spread} spread, {off} offset");
        assert!(spread < len, "otherwise this just tests memcpy");
        let mut buf = AlignedSlice::new_zeroed(len + spread, off);
        let mut fill: usize = 0;
        buf[..len].fill_with(|| {
            fill += 1;
            fill as u8
        });
        (len, spread, buf)
    }

    fn setup_backward(cfg: Cfg) -> (usize, usize, AlignedSlice) {
        let Cfg { len, spread, off } = cfg;
        let spread = calculate_spread(len, spread);
        println!("{len} bytes, {spread} spread, {off} offset");
        assert!(spread < len, "otherwise this just tests memcpy");
        let mut buf = AlignedSlice::new_zeroed(len + spread, off);
        let mut fill: usize = 0;
        buf[spread..].fill_with(|| {
            fill += 1;
            fill as u8
        });
        (len, spread, buf)
    }

    #[library_benchmark]
    #[benches::small_spread(
        args = [
            Cfg { len: 16, spread: Small, off: 0 },
            Cfg { len: 32, spread: Small, off: 0 },
            Cfg { len: 64, spread: Small, off: 0 },
            Cfg { len: 512, spread: Small, off: 0 },
            Cfg { len: 4096, spread: Small, off: 0 },
            Cfg { len: 1048576, spread: Small, off: 0 },
        ],
        setup = setup_forward
    )]
    #[benches::medium_spread(
        args = [
            Cfg { len: 16, spread: Medium, off: 0 },
            Cfg { len: 32, spread: Medium, off: 0 },
            Cfg { len: 64, spread: Medium, off: 0 },
            Cfg { len: 512, spread: Medium, off: 0 },
            Cfg { len: 4096, spread: Medium, off: 0 },
            Cfg { len: 1048576, spread: Medium, off: 0 },
        ],
        setup = setup_forward
    )]
    #[benches::large_spread(
        args = [
            Cfg { len: 16, spread: Large, off: 0 },
            Cfg { len: 32, spread: Large, off: 0 },
            Cfg { len: 64, spread: Large, off: 0 },
            Cfg { len: 512, spread: Large, off: 0 },
            Cfg { len: 4096, spread: Large, off: 0 },
            Cfg { len: 1048576, spread: Large, off: 0 },
        ],
        setup = setup_forward
    )]
    #[benches::small_spread_offset(
        args = [
            Cfg { len: 16, spread: Small, off: 63 },
            Cfg { len: 32, spread: Small, off: 63 },
            Cfg { len: 64, spread: Small, off: 63 },
            Cfg { len: 512, spread: Small, off: 63 },
            Cfg { len: 4096, spread: Small, off: 63 },
            Cfg { len: 1048576, spread: Small, off: 63 },
        ],
        setup = setup_forward
    )]
    #[benches::medium_spread_offset(
        args = [
            Cfg { len: 16, spread: Medium, off: 63 },
            Cfg { len: 32, spread: Medium, off: 63 },
            Cfg { len: 64, spread: Medium, off: 63 },
            Cfg { len: 512, spread: Medium, off: 63 },
            Cfg { len: 4096, spread: Medium, off: 63 },
            Cfg { len: 1048576, spread: Medium, off: 63 },
        ],
        setup = setup_forward
    )]
    #[benches::large_spread_offset(
        args = [
            Cfg { len: 16, spread: Large, off: 63 },
            Cfg { len: 32, spread: Large, off: 63 },
            Cfg { len: 64, spread: Large, off: 63 },
            Cfg { len: 512, spread: Large, off: 63 },
            Cfg { len: 4096, spread: Large, off: 63 },
            Cfg { len: 1048576, spread: Large, off: 63 },
        ],
        setup = setup_forward
    )]
    fn forward((len, spread, mut buf): (usize, usize, AlignedSlice)) {
        // Test moving from the start of the buffer toward the end
        unsafe {
            black_box(memmove(
                black_box(buf[spread..].as_mut_ptr()),
                black_box(buf.as_ptr()),
                black_box(len),
            ));
        }
    }

    #[library_benchmark]
    #[benches::small_spread(
        args = [
            Cfg { len: 16, spread: Small, off: 0 },
            Cfg { len: 32, spread: Small, off: 0 },
            Cfg { len: 64, spread: Small, off: 0 },
            Cfg { len: 512, spread: Small, off: 0 },
            Cfg { len: 4096, spread: Small, off: 0 },
            Cfg { len: 1048576, spread: Small, off: 0 },
        ],
        setup = setup_backward
    )]
    #[benches::middle(
        args = [
            Cfg { len: 16, spread: Medium, off: 0 },
            Cfg { len: 32, spread: Medium, off: 0 },
            Cfg { len: 64, spread: Medium, off: 0 },
            Cfg { len: 512, spread: Medium, off: 0 },
            Cfg { len: 4096, spread: Medium, off: 0 },
            Cfg { len: 1048576, spread: Medium, off: 0 },
        ],
        setup = setup_backward
    )]
    #[benches::large_spread(
        args = [
            Cfg { len: 16, spread: Large, off: 0 },
            Cfg { len: 32, spread: Large, off: 0 },
            Cfg { len: 64, spread: Large, off: 0 },
            Cfg { len: 512, spread: Large, off: 0 },
            Cfg { len: 4096, spread: Large, off: 0 },
            Cfg { len: 1048576, spread: Large, off: 0 },
        ],
        setup = setup_backward
    )]
    #[benches::small_spread_off(
        args = [
            Cfg { len: 16, spread: Small, off: 63 },
            Cfg { len: 32, spread: Small, off: 63 },
            Cfg { len: 64, spread: Small, off: 63 },
            Cfg { len: 512, spread: Small, off: 63 },
            Cfg { len: 4096, spread: Small, off: 63 },
            Cfg { len: 1048576, spread: Small, off: 63 },
        ],
        setup = setup_backward
    )]
    #[benches::middle_off(
        args = [
            Cfg { len: 16, spread: Medium, off: 63 },
            Cfg { len: 32, spread: Medium, off: 63 },
            Cfg { len: 64, spread: Medium, off: 63 },
            Cfg { len: 512, spread: Medium, off: 63 },
            Cfg { len: 4096, spread: Medium, off: 63 },
            Cfg { len: 1048576, spread: Medium, off: 63 },
        ],
        setup = setup_backward
    )]
    #[benches::large_spread_off(
        args = [
            Cfg { len: 16, spread: Large, off: 63 },
            Cfg { len: 32, spread: Large, off: 63 },
            Cfg { len: 64, spread: Large, off: 63 },
            Cfg { len: 512, spread: Large, off: 63 },
            Cfg { len: 4096, spread: Large, off: 63 },
            Cfg { len: 1048576, spread: Large, off: 63 },
        ],
        setup = setup_backward
    )]
    fn backward((len, spread, mut buf): (usize, usize, AlignedSlice)) {
        // Test moving from the end of the buffer toward the start
        unsafe {
            black_box(memmove(
                black_box(buf.as_mut_ptr()),
                black_box(buf[spread..].as_ptr()),
                black_box(len),
            ));
        }
    }

    library_benchmark_group!(name = memmove; benchmarks = forward, backward);
}

use mcmp::memcmp;
use mcpy::memcpy;
use mmove::memmove;
use mset::memset;

main!(library_benchmark_groups = memcpy, memset, memcmp, memmove);
