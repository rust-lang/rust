//! Benchmarks that use Callgrind (via `gungraun`) to report instruction count metrics. This
//! is stable enough to be tested in CI.

use std::hint::black_box;

use builtins_test::mem::{AlignedSlice, MEG1};
use gungraun::{library_benchmark, library_benchmark_group, main};

mod mcpy {
    use builtins_test::mem::mcpy::{Cfg, setup};
    use compiler_builtins::mem::memcpy;

    use super::*;

    #[library_benchmark]
    #[benches::aligned(
        // Both aligned
        args = [
            Cfg { len: 16, s_off: 0, d_off: 0 },
            Cfg { len: 32, s_off: 0, d_off: 0 },
            Cfg { len: 64, s_off: 0, d_off: 0 },
            Cfg { len: 512, s_off: 0, d_off: 0 },
            Cfg { len: 4096, s_off: 0, d_off: 0 },
            Cfg { len: MEG1, s_off: 0, d_off: 0 },
        ],
        setup = setup,
    )]
    #[benches::offset(
        // Both unaligned but at the same offset
        args = [
            Cfg { len: 16, s_off: 65, d_off: 65 },
            Cfg { len: 32, s_off: 65, d_off: 65 },
            Cfg { len: 64, s_off: 65, d_off: 65 },
            Cfg { len: 512, s_off: 65, d_off: 65 },
            Cfg { len: 4096, s_off: 65, d_off: 65 },
            Cfg { len: MEG1, s_off: 65, d_off: 65 },
        ],
        setup = setup,
    )]
    #[benches::misaligned(
        // `src` and `dst` both misaligned by different amounts
        args = [
            Cfg { len: 16, s_off: 65, d_off: 66 },
            Cfg { len: 32, s_off: 65, d_off: 66 },
            Cfg { len: 64, s_off: 65, d_off: 66 },
            Cfg { len: 512, s_off: 65, d_off: 66 },
            Cfg { len: 4096, s_off: 65, d_off: 66 },
            Cfg { len: MEG1, s_off: 65, d_off: 66 },
        ],
        setup = setup,
    )]
    fn bench_cpy((len, mut dst, src): (usize, AlignedSlice, AlignedSlice)) {
        unsafe {
            black_box(memcpy(
                black_box(dst.as_mut_ptr()),
                black_box(src.as_ptr()),
                black_box(len),
            ));
        }
    }

    library_benchmark_group!(name = memcpy, benchmarks = [bench_cpy]);
}

mod mset {
    use builtins_test::mem::mset::{Cfg, setup};
    use compiler_builtins::mem::memset;

    use super::*;

    #[library_benchmark]
    #[benches::aligned(
        args = [
            Cfg { len: 16, offset: 0 },
            Cfg { len: 32, offset: 0 },
            Cfg { len: 64, offset: 0 },
            Cfg { len: 512, offset: 0 },
            Cfg { len: 4096, offset: 0 },
            Cfg { len: MEG1, offset: 0 },
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
            Cfg { len: MEG1, offset: 65 },
        ],
        setup = setup,
    )]
    fn bench_set((len, mut dst): (usize, AlignedSlice)) {
        unsafe {
            black_box(memset(
                black_box(dst.as_mut_ptr()),
                black_box(27),
                black_box(len),
            ));
        }
    }

    library_benchmark_group!(name = memset, benchmarks = [bench_set]);
}

mod mcmp {
    use builtins_test::mem::mcmp::{Cfg, setup};
    use compiler_builtins::mem::memcmp;

    use super::*;

    #[library_benchmark]
    #[benches::aligned(
        // Both aligned
        args = [
            Cfg { len: 16, s_off: 0, d_off: 0 },
            Cfg { len: 32, s_off: 0, d_off: 0 },
            Cfg { len: 64, s_off: 0, d_off: 0 },
            Cfg { len: 512, s_off: 0, d_off: 0 },
            Cfg { len: 4096, s_off: 0, d_off: 0 },
            Cfg { len: MEG1, s_off: 0, d_off: 0 },
        ],
        setup = setup
    )]
    #[benches::offset(
        // Both at the same offset
        args = [
            Cfg { len: 16, s_off: 65, d_off: 65 },
            Cfg { len: 32, s_off: 65, d_off: 65 },
            Cfg { len: 64, s_off: 65, d_off: 65 },
            Cfg { len: 512, s_off: 65, d_off: 65 },
            Cfg { len: 4096, s_off: 65, d_off: 65 },
            Cfg { len: MEG1, s_off: 65, d_off: 65 },
        ],
        setup = setup
    )]
    #[benches::misaligned(
        // `src` and `dst` both misaligned by different amounts
        args = [
            Cfg { len: 16, s_off: 65, d_off: 66 },
            Cfg { len: 32, s_off: 65, d_off: 66 },
            Cfg { len: 64, s_off: 65, d_off: 66 },
            Cfg { len: 512, s_off: 65, d_off: 66 },
            Cfg { len: 4096, s_off: 65, d_off: 66 },
            Cfg { len: MEG1, s_off: 65, d_off: 66 },
        ],
        setup = setup
    )]
    fn bench_cmp((len, mut dst, src): (usize, AlignedSlice, AlignedSlice)) {
        unsafe {
            black_box(memcmp(
                black_box(dst.as_mut_ptr()),
                black_box(src.as_ptr()),
                black_box(len),
            ));
        }
    }

    library_benchmark_group!(name = memcmp, benchmarks = [bench_cmp]);
}

mod mmove {
    use Spread::{Aligned, Large, Medium, Small};
    use builtins_test::mem::mmove::{Cfg, Spread, setup_backward, setup_forward};
    use compiler_builtins::mem::memmove;

    use super::*;

    #[library_benchmark]
    #[benches::aligned(
        args = [
            // Don't test small spreads since there is no overlap
            Cfg { len: 4096, spread: Aligned, off: 0 },
            Cfg { len: MEG1, spread: Aligned, off: 0 },
        ],
        setup = setup_forward
    )]
    #[benches::small_spread(
        args = [
            Cfg { len: 16, spread: Small, off: 0 },
            Cfg { len: 32, spread: Small, off: 0 },
            Cfg { len: 64, spread: Small, off: 0 },
            Cfg { len: 512, spread: Small, off: 0 },
            Cfg { len: 4096, spread: Small, off: 0 },
            Cfg { len: MEG1, spread: Small, off: 0 },
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
            Cfg { len: MEG1, spread: Medium, off: 0 },
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
            Cfg { len: MEG1, spread: Large, off: 0 },
        ],
        setup = setup_forward
    )]
    #[benches::aligned_off(
        args = [
            Cfg { len: 4096, spread: Aligned, off: 65 },
            Cfg { len: MEG1, spread: Aligned, off: 65 },
        ],
        setup = setup_forward
    )]
    #[benches::small_spread_off(
        args = [
            Cfg { len: 16, spread: Small, off: 65 },
            Cfg { len: 32, spread: Small, off: 65 },
            Cfg { len: 64, spread: Small, off: 65 },
            Cfg { len: 512, spread: Small, off: 65 },
            Cfg { len: 4096, spread: Small, off: 65 },
            Cfg { len: MEG1, spread: Small, off: 65 },
        ],
        setup = setup_forward
    )]
    #[benches::medium_spread_off(
        args = [
            Cfg { len: 16, spread: Medium, off: 65 },
            Cfg { len: 32, spread: Medium, off: 65 },
            Cfg { len: 64, spread: Medium, off: 65 },
            Cfg { len: 512, spread: Medium, off: 65 },
            Cfg { len: 4096, spread: Medium, off: 65 },
            Cfg { len: MEG1, spread: Medium, off: 65 },
        ],
        setup = setup_forward
    )]
    #[benches::large_spread_off(
        args = [
            Cfg { len: 16, spread: Large, off: 65 },
            Cfg { len: 32, spread: Large, off: 65 },
            Cfg { len: 64, spread: Large, off: 65 },
            Cfg { len: 512, spread: Large, off: 65 },
            Cfg { len: 4096, spread: Large, off: 65 },
            Cfg { len: MEG1, spread: Large, off: 65 },
        ],
        setup = setup_forward
    )]
    fn forward_move((len, spread, mut buf): (usize, usize, AlignedSlice)) {
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
    #[benches::aligned(
        args = [
            // Don't test small spreads since there is no overlap
            Cfg { len: 4096, spread: Aligned, off: 0 },
            Cfg { len: MEG1, spread: Aligned, off: 0 },
        ],
        setup = setup_backward
    )]
    #[benches::small_spread(
        args = [
            Cfg { len: 16, spread: Small, off: 0 },
            Cfg { len: 32, spread: Small, off: 0 },
            Cfg { len: 64, spread: Small, off: 0 },
            Cfg { len: 512, spread: Small, off: 0 },
            Cfg { len: 4096, spread: Small, off: 0 },
            Cfg { len: MEG1, spread: Small, off: 0 },
        ],
        setup = setup_backward
    )]
    #[benches::medium_spread(
        args = [
            Cfg { len: 16, spread: Medium, off: 0 },
            Cfg { len: 32, spread: Medium, off: 0 },
            Cfg { len: 64, spread: Medium, off: 0 },
            Cfg { len: 512, spread: Medium, off: 0 },
            Cfg { len: 4096, spread: Medium, off: 0 },
            Cfg { len: MEG1, spread: Medium, off: 0 },
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
            Cfg { len: MEG1, spread: Large, off: 0 },
        ],
        setup = setup_backward
    )]
    #[benches::aligned_off(
        args = [
            // Don't test small spreads since there is no overlap
            Cfg { len: 4096, spread: Aligned, off: 65 },
            Cfg { len: MEG1, spread: Aligned, off: 65 },
        ],
        setup = setup_backward
    )]
    #[benches::small_spread_off(
        args = [
            Cfg { len: 16, spread: Small, off: 65 },
            Cfg { len: 32, spread: Small, off: 65 },
            Cfg { len: 64, spread: Small, off: 65 },
            Cfg { len: 512, spread: Small, off: 65 },
            Cfg { len: 4096, spread: Small, off: 65 },
            Cfg { len: MEG1, spread: Small, off: 65 },
        ],
        setup = setup_backward
    )]
    #[benches::medium_spread_off(
        args = [
            Cfg { len: 16, spread: Medium, off: 65 },
            Cfg { len: 32, spread: Medium, off: 65 },
            Cfg { len: 64, spread: Medium, off: 65 },
            Cfg { len: 512, spread: Medium, off: 65 },
            Cfg { len: 4096, spread: Medium, off: 65 },
            Cfg { len: MEG1, spread: Medium, off: 65 },
        ],
        setup = setup_backward
    )]
    #[benches::large_spread_off(
        args = [
            Cfg { len: 16, spread: Large, off: 65 },
            Cfg { len: 32, spread: Large, off: 65 },
            Cfg { len: 64, spread: Large, off: 65 },
            Cfg { len: 512, spread: Large, off: 65 },
            Cfg { len: 4096, spread: Large, off: 65 },
            Cfg { len: MEG1, spread: Large, off: 65 },
        ],
        setup = setup_backward
    )]
    fn backward_move((len, spread, mut buf): (usize, usize, AlignedSlice)) {
        // Test moving from the end of the buffer toward the start
        unsafe {
            black_box(memmove(
                black_box(buf.as_mut_ptr()),
                black_box(buf[spread..].as_ptr()),
                black_box(len),
            ));
        }
    }

    library_benchmark_group!(name = memmove, benchmarks = [forward_move, backward_move]);
}

mod slen {
    use builtins_test::mem::slen::{Cfg, setup};
    use compiler_builtins::mem::strlen;

    use super::*;

    #[library_benchmark]
    #[benches::aligned(
        args = [
            Cfg { len: 1, offset: 0 },
            Cfg { len: 16, offset: 0 },
            Cfg { len: 32, offset: 0 },
            Cfg { len: 64, offset: 0 },
            Cfg { len: 512, offset: 0 },
            Cfg { len: 4096, offset: 0 },
            Cfg { len: MEG1, offset: 0 },
        ],
        setup = setup,
    )]
    #[benches::offset(
        args = [
            Cfg { len: 1, offset: 65 },
            Cfg { len: 16, offset: 65 },
            Cfg { len: 32, offset: 65 },
            Cfg { len: 64, offset: 65 },
            Cfg { len: 512, offset: 65 },
            Cfg { len: 4096, offset: 65 },
            Cfg { len: MEG1, offset: 65 },
        ],
        setup = setup,
    )]
    fn bench_strlen(s: AlignedSlice) {
        unsafe {
            black_box(strlen(black_box(s.as_ptr().cast::<core::ffi::c_char>())));
        }
    }

    library_benchmark_group!(name = strlen, benchmarks = [bench_strlen]);
}

use mcmp::memcmp;
use mcpy::memcpy;
use mmove::memmove;
use mset::memset;
use slen::strlen;

main!(library_benchmark_groups = [memcpy, memset, memcmp, memmove, strlen]);
