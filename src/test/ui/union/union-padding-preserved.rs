// run-pass

// Test that unions don't lose padding bytes (i.e. not covered by any leaf field).

#![feature(core_intrinsics, test, transparent_unions)]

extern crate test;

#[repr(transparent)]
#[derive(Copy, Clone)]
union U<T: Copy> { _x: T, y: () }

impl<T: Copy> U<T> {
    fn uninit() -> Self {
        U { y: () }
    }

    unsafe fn write(&mut self, i: usize, v: u8) {
        (self as *mut _ as *mut u8).add(i).write(v);
    }

    #[inline(never)]
    unsafe fn read_rust(self, i: usize) -> u8 {
        test::black_box((&self as *const _ as *const u8).add(i)).read()
    }

    #[inline(never)]
    unsafe extern "C" fn read_c(self, i: usize) -> u8 {
        test::black_box((&self as *const _ as *const u8).add(i)).read()
    }
}

#[derive(Copy, Clone, Default)]
struct Options {
    demote_c_to_warning: bool,
}

unsafe fn check_at<T: Copy>(i: usize, v: u8, opts: Options) {
    let mut u = U::<T>::uninit();
    u.write(i, v);
    let msg = |abi: &str| format!(
        "check_at::<{}>: {} ABI failed at byte {}",
        std::intrinsics::type_name::<T>(),
        abi,
        i,
    );
    if u.read_rust(i) != v {
        panic!(msg("Rust"));
    }
    if u.read_c(i) != v {
        if opts.demote_c_to_warning {
            eprintln!("warning: {}", msg("C"));
        } else {
            panic!(msg("C"));
        }
    }
}

fn check_all<T: Copy>(opts: Options) {
    for i in 0..std::mem::size_of::<T>() {
        unsafe {
            check_at::<T>(i, 100 + i as u8, opts);
        }
    }
}

#[repr(C, align(16))]
#[derive(Copy, Clone)]
struct Align16<T>(T);

#[repr(C)]
#[derive(Copy, Clone)]
struct Pair<A, B>(A, B);

fn main() {
    // NOTE(eddyb) we can't error for the `extern "C"` ABI in these cases, as
    // the x86_64 SysV calling convention will drop padding bytes inside unions.
    check_all::<Align16<u8>>(Options { demote_c_to_warning: true });
    check_all::<Align16<f32>>(Options { demote_c_to_warning: true });
    check_all::<Align16<f64>>(Options { demote_c_to_warning: true });

    check_all::<(u8, u32)>(Options::default());
    check_all::<(u8, u64)>(Options::default());

    check_all::<Pair<u8, u32>>(Options::default());
    check_all::<Pair<u8, u64>>(Options::default());
}
