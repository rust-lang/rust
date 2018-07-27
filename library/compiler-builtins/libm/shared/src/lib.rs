#![feature(exact_chunks)]

#[macro_use]
extern crate lazy_static;

lazy_static! {
    pub static ref F32: Vec<f32> = {
        let bytes = include_bytes!("../../bin/input/f32");

        bytes
            .exact_chunks(4)
            .map(|chunk| {
                let mut buf = [0; 4];
                buf.copy_from_slice(chunk);
                f32::from_bits(u32::from_le(u32::from_bytes(buf)))
            })
            .collect()
    };
    pub static ref F32F32: Vec<(f32, f32)> = {
        let bytes = include_bytes!("../../bin/input/f32f32");

        bytes
            .exact_chunks(8)
            .map(|chunk| {
                let mut x0 = [0; 4];
                let mut x1 = [0; 4];
                x0.copy_from_slice(&chunk[..4]);
                x1.copy_from_slice(&chunk[4..]);

                (
                    f32::from_bits(u32::from_le(u32::from_bytes(x0))),
                    f32::from_bits(u32::from_le(u32::from_bytes(x1))),
                )
            })
            .collect()
    };
    pub static ref F32F32F32: Vec<(f32, f32, f32)> = {
        let bytes = include_bytes!("../../bin/input/f32f32f32");

        bytes
            .exact_chunks(12)
            .map(|chunk| {
                let mut x0 = [0; 4];
                let mut x1 = [0; 4];
                let mut x2 = [0; 4];
                x0.copy_from_slice(&chunk[..4]);
                x1.copy_from_slice(&chunk[4..8]);
                x2.copy_from_slice(&chunk[8..]);

                (
                    f32::from_bits(u32::from_le(u32::from_bytes(x0))),
                    f32::from_bits(u32::from_le(u32::from_bytes(x1))),
                    f32::from_bits(u32::from_le(u32::from_bytes(x2))),
                )
            })
            .collect()
    };
    pub static ref F32I32: Vec<(f32, i32)> = {
        let bytes = include_bytes!("../../bin/input/f32i16");

        bytes
            .exact_chunks(6)
            .map(|chunk| {
                let mut x0 = [0; 4];
                let mut x1 = [0; 2];
                x0.copy_from_slice(&chunk[..4]);
                x1.copy_from_slice(&chunk[4..]);

                (
                    f32::from_bits(u32::from_le(u32::from_bytes(x0))),
                    i16::from_le(i16::from_bytes(x1)) as i32,
                )
            })
            .collect()
    };
    pub static ref F64: Vec<f64> = {
        let bytes = include_bytes!("../../bin/input/f64");

        bytes
            .exact_chunks(8)
            .map(|chunk| {
                let mut buf = [0; 8];
                buf.copy_from_slice(chunk);
                f64::from_bits(u64::from_le(u64::from_bytes(buf)))
            })
            .collect()
    };
    pub static ref F64F64: Vec<(f64, f64)> = {
        let bytes = include_bytes!("../../bin/input/f64f64");

        bytes
            .exact_chunks(16)
            .map(|chunk| {
                let mut x0 = [0; 8];
                let mut x1 = [0; 8];
                x0.copy_from_slice(&chunk[..8]);
                x1.copy_from_slice(&chunk[8..]);

                (
                    f64::from_bits(u64::from_le(u64::from_bytes(x0))),
                    f64::from_bits(u64::from_le(u64::from_bytes(x1))),
                )
            })
            .collect()
    };
    pub static ref F64F64F64: Vec<(f64, f64, f64)> = {
        let bytes = include_bytes!("../../bin/input/f64f64f64");

        bytes
            .exact_chunks(24)
            .map(|chunk| {
                let mut x0 = [0; 8];
                let mut x1 = [0; 8];
                let mut x2 = [0; 8];
                x0.copy_from_slice(&chunk[..8]);
                x1.copy_from_slice(&chunk[8..16]);
                x2.copy_from_slice(&chunk[16..]);

                (
                    f64::from_bits(u64::from_le(u64::from_bytes(x0))),
                    f64::from_bits(u64::from_le(u64::from_bytes(x1))),
                    f64::from_bits(u64::from_le(u64::from_bytes(x2))),
                )
            })
            .collect()
    };
    pub static ref F64I32: Vec<(f64, i32)> = {
        let bytes = include_bytes!("../../bin/input/f64i16");

        bytes
            .exact_chunks(10)
            .map(|chunk| {
                let mut x0 = [0; 8];
                let mut x1 = [0; 2];
                x0.copy_from_slice(&chunk[..8]);
                x1.copy_from_slice(&chunk[8..]);

                (
                    f64::from_bits(u64::from_le(u64::from_bytes(x0))),
                    i16::from_le(i16::from_bytes(x1)) as i32,
                )
            })
            .collect()
    };
}

#[macro_export]
macro_rules! f32 {
    ($lib:expr, $($fun:ident),+) => {
        $(
            #[test]
            fn $fun() {
                let expected = include_bytes!(concat!("../bin/output/", $lib, ".", stringify!($fun)))
                    .exact_chunks(4)
                    .map(|chunk| {
                        let mut buf = [0; 4];
                        buf.copy_from_slice(chunk);
                        f32::from_bits(u32::from_le(u32::from_bytes(buf)))
                    })
                    .collect::<Vec<_>>();

                for (input, expected) in $crate::F32.iter().zip(&expected) {
                    if let Ok(output) = panic::catch_unwind(|| libm::$fun(*input)) {
                        if let Err(error) = libm::_eqf(output, *expected) {
                            panic!(
                                "INPUT: {:#x}, OUTPUT: {:#x}, EXPECTED: {:#x}, ERROR: {}",
                                input.to_bits(),
                                output.to_bits(),
                                expected.to_bits(),
                                error
                            );
                        }
                    } else {
                        panic!(
                            "INPUT: {:#x}, OUTPUT: PANIC!, EXPECTED: {:#x}",
                            input.to_bits(),
                            expected.to_bits()
                        );
                    }
                }
            }
        )+
    }
}

#[macro_export]
macro_rules! f32f32 {
    ($lib:expr, $($fun:ident),+) => {
        $(
            #[test]
            fn $fun() {
                let expected = include_bytes!(concat!("../bin/output/", $lib, ".", stringify!($fun)))
                    .exact_chunks(4)
                    .map(|chunk| {
                        let mut buf = [0; 4];
                        buf.copy_from_slice(chunk);
                        f32::from_bits(u32::from_le(u32::from_bytes(buf)))
                    })
                    .collect::<Vec<_>>();

                for ((i0, i1), expected) in $crate::F32F32.iter().zip(&expected) {
                    if let Ok(output) = panic::catch_unwind(|| libm::$fun(*i0, *i1)) {
                        if let Err(error) = libm::_eqf(output, *expected) {
                            panic!(
                                "INPUT: ({:#x}, {:#x}), OUTPUT: {:#x}, EXPECTED: {:#x}, ERROR: {}",
                                i0.to_bits(),
                                i1.to_bits(),
                                output.to_bits(),
                                expected.to_bits(),
                                error
                            );
                        }
                    } else {
                        panic!(
                            "INPUT: ({:#x}, {:#x}), OUTPUT: PANIC!, EXPECTED: {:#x}",
                            i0.to_bits(),
                            i1.to_bits(),
                            expected.to_bits()
                        );
                    }
                }
            }
        )+
    }
}

#[macro_export]
macro_rules! f32f32f32 {
    ($lib:expr, $($fun:ident),+) => {
        $(
            #[test]
            fn $fun() {
                let expected = include_bytes!(concat!("../bin/output/", $lib, ".", stringify!($fun)))
                    .exact_chunks(4)
                    .map(|chunk| {
                        let mut buf = [0; 4];
                        buf.copy_from_slice(chunk);
                        f32::from_bits(u32::from_le(u32::from_bytes(buf)))
                    })
                    .collect::<Vec<_>>();

                for ((i0, i1, i2), expected) in $crate::F32F32F32.iter().zip(&expected) {
                    if let Ok(output) = panic::catch_unwind(|| libm::$fun(*i0, *i1, *i2)) {
                        if let Err(error) = libm::_eqf(output, *expected) {
                            panic!(
                                "INPUT: ({:#x}, {:#x}, {:#x}), OUTPUT: {:#x}, EXPECTED: {:#x}, ERROR: {}",
                                i0.to_bits(),
                                i1.to_bits(),
                                i2.to_bits(),
                                output.to_bits(),
                                expected.to_bits(),
                                error
                            );
                        }
                    } else {
                        panic!(
                            "INPUT: ({:#x}, {:#x}), OUTPUT: PANIC!, EXPECTED: {:#x}",
                            i0.to_bits(),
                            i1.to_bits(),
                            expected.to_bits()
                        );
                    }
                }
            }
        )+
    }
}

#[macro_export]
macro_rules! f32i32 {
    ($lib:expr, $($fun:ident),+) => {
        $(
            #[test]
            fn $fun() {
                let expected = include_bytes!(concat!("../bin/output/", $lib, ".", stringify!($fun)))
                    .exact_chunks(4)
                    .map(|chunk| {
                        let mut buf = [0; 4];
                        buf.copy_from_slice(chunk);
                        f32::from_bits(u32::from_le(u32::from_bytes(buf)))
                    })
                    .collect::<Vec<_>>();

                for ((i0, i1), expected) in $crate::F32I32.iter().zip(&expected) {
                    if let Ok(output) = panic::catch_unwind(|| libm::$fun(*i0, *i1)) {
                        if let Err(error) = libm::_eqf(output, *expected) {
                            panic!(
                                "INPUT: ({:#x}, {:#x}), OUTPUT: {:#x}, EXPECTED: {:#x}, ERROR: {}",
                                i0.to_bits(),
                                i1,
                                output.to_bits(),
                                expected.to_bits(),
                                error
                            );
                        }
                    } else {
                        panic!(
                            "INPUT: ({:#x}, {:#x}), OUTPUT: PANIC!, EXPECTED: {:#x}",
                            i0.to_bits(),
                            i1,
                            expected.to_bits()
                        );
                    }
                }
            }
        )+
    }
}

#[macro_export]
macro_rules! f64 {
    ($lib:expr, $($fun:ident),+) => {
        $(
            #[test]
            fn $fun() {
                let expected = include_bytes!(concat!("../bin/output/", $lib, ".", stringify!($fun)))
                    .exact_chunks(8)
                    .map(|chunk| {
                        let mut buf = [0; 8];
                        buf.copy_from_slice(chunk);
                        f64::from_bits(u64::from_le(u64::from_bytes(buf)))
                    })
                    .collect::<Vec<_>>();

                for (input, expected) in shared::F64.iter().zip(&expected) {
                    if let Ok(output) = panic::catch_unwind(|| libm::$fun(*input)) {
                        if let Err(error) = libm::_eq(output, *expected) {
                            panic!(
                                "INPUT: {:#x}, OUTPUT: {:#x}, EXPECTED: {:#x}, ERROR: {}",
                                input.to_bits(),
                                output.to_bits(),
                                expected.to_bits(),
                                error
                            );
                        }
                    } else {
                        panic!(
                            "INPUT: {:#x}, OUTPUT: PANIC!, EXPECTED: {:#x}",
                            input.to_bits(),
                            expected.to_bits()
                        );
                    }
                }
            }
        )+
    }
}

#[macro_export]
macro_rules! f64f64 {
    ($lib:expr, $($fun:ident),+) => {
        $(
            #[test]
            fn $fun() {
                let expected = include_bytes!(concat!("../bin/output/", $lib, ".", stringify!($fun)))
                    .exact_chunks(8)
                    .map(|chunk| {
                        let mut buf = [0; 8];
                        buf.copy_from_slice(chunk);
                        f64::from_bits(u64::from_le(u64::from_bytes(buf)))
                    })
                    .collect::<Vec<_>>();

                for ((i0, i1), expected) in shared::F64F64.iter().zip(&expected) {
                    if let Ok(output) = panic::catch_unwind(|| libm::$fun(*i0, *i1)) {
                        if let Err(error) = libm::_eq(output, *expected) {
                            panic!(
                                "INPUT: ({:#x}, {:#x}), OUTPUT: {:#x}, EXPECTED: {:#x}, ERROR: {}",
                                i0.to_bits(),
                                i1.to_bits(),
                                output.to_bits(),
                                expected.to_bits(),
                                error
                            );
                        }
                    } else {
                        panic!(
                            "INPUT: ({:#x}, {:#x}), OUTPUT: PANIC!, EXPECTED: {:#x}",
                            i0.to_bits(),
                            i1.to_bits(),
                            expected.to_bits()
                        );
                    }
                }
            }
        )+
    }
}

#[macro_export]
macro_rules! f64f64f64 {
    ($lib:expr, $($fun:ident),+) => {
        $(
            #[test]
            fn $fun() {
                let expected = include_bytes!(concat!("../bin/output/", $lib, ".", stringify!($fun)))
                    .exact_chunks(8)
                    .map(|chunk| {
                        let mut buf = [0; 8];
                        buf.copy_from_slice(chunk);
                        f64::from_bits(u64::from_le(u64::from_bytes(buf)))
                    })
                    .collect::<Vec<_>>();

                for ((i0, i1, i2), expected) in shared::F64F64F64.iter().zip(&expected) {
                    if let Ok(output) = panic::catch_unwind(|| libm::$fun(*i0, *i1, *i2)) {
                        if let Err(error) = libm::_eq(output, *expected) {
                            panic!(
                                "INPUT: ({:#x}, {:#x}, {:#x}), OUTPUT: {:#x}, EXPECTED: {:#x}, ERROR: {}",
                                i0.to_bits(),
                                i1.to_bits(),
                                i2.to_bits(),
                                output.to_bits(),
                                expected.to_bits(),
                                error
                            );
                        }
                    } else {
                        panic!(
                            "INPUT: ({:#x}, {:#x}), OUTPUT: PANIC!, EXPECTED: {:#x}",
                            i0.to_bits(),
                            i1.to_bits(),
                            expected.to_bits()
                        );
                    }
                }
            }
        )+
    }
}

#[macro_export]
macro_rules! f64i32 {
    ($lib:expr, $($fun:ident),+) => {
        $(
            #[test]
            fn $fun() {
                let expected = include_bytes!(concat!("../bin/output/", $lib, ".", stringify!($fun)))
                    .exact_chunks(8)
                    .map(|chunk| {
                        let mut buf = [0; 8];
                        buf.copy_from_slice(chunk);
                        f64::from_bits(u64::from_le(u64::from_bytes(buf)))
                    })
                    .collect::<Vec<_>>();

                for ((i0, i1), expected) in shared::F64I32.iter().zip(&expected) {
                    if let Ok(output) = panic::catch_unwind(|| libm::$fun(*i0, *i1)) {
                        if let Err(error) = libm::_eq(output, *expected) {
                            panic!(
                                "INPUT: ({:#x}, {:#x}), OUTPUT: {:#x}, EXPECTED: {:#x}, ERROR: {}",
                                i0.to_bits(),
                                i1,
                                output.to_bits(),
                                expected.to_bits(),
                                error
                            );
                        }
                    } else {
                        panic!(
                            "INPUT: ({:#x}, {:#x}), OUTPUT: PANIC!, EXPECTED: {:#x}",
                            i0.to_bits(),
                            i1,
                            expected.to_bits()
                        );
                    }
                }
            }
        )+
    }
}
