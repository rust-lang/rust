use rand::seq::SliceRandom;
use rand::Rng;
use std::collections::HashMap;
use std::fmt;
use std::fmt::Write as FmtWrite;
use std::fs::{self, OpenOptions};
use std::hash::{Hash, Hasher};
use std::io::Write;
use std::path::PathBuf;
use std::{env, mem};

const NTESTS: usize = 1_000;

fn main() {
    let out_dir = PathBuf::from(env::var_os("OUT_DIR").unwrap());
    let out_file = out_dir.join("generated.rs");
    drop(fs::remove_file(&out_file));

    let target = env::var("TARGET").unwrap();
    let target_arch_arm = target.contains("arm") || target.contains("thumb");
    let target_arch_mips = target.contains("mips");

    // TODO accept NaNs. We don't do that right now because we can't check
    // for NaN-ness on the thumb targets (due to missing intrinsics)

    // float/add.rs
    gen(
        |(a, b): (MyF64, MyF64)| {
            let c = a.0 + b.0;
            if a.0.is_nan() || b.0.is_nan() || c.is_nan() {
                None
            } else {
                Some(c)
            }
        },
        "builtins::float::add::__adddf3(a, b)",
    );
    gen(
        |(a, b): (MyF32, MyF32)| {
            let c = a.0 + b.0;
            if a.0.is_nan() || b.0.is_nan() || c.is_nan() {
                None
            } else {
                Some(c)
            }
        },
        "builtins::float::add::__addsf3(a, b)",
    );

    if target_arch_arm {
        gen(
            |(a, b): (MyF64, MyF64)| {
                let c = a.0 + b.0;
                if a.0.is_nan() || b.0.is_nan() || c.is_nan() {
                    None
                } else {
                    Some(c)
                }
            },
            "builtins::float::add::__adddf3vfp(a, b)",
        );
        gen(
            |(a, b): (LargeF32, LargeF32)| {
                let c = a.0 + b.0;
                if a.0.is_nan() || b.0.is_nan() || c.is_nan() {
                    None
                } else {
                    Some(c)
                }
            },
            "builtins::float::add::__addsf3vfp(a, b)",
        );
    }

    // float/cmp.rs
    gen(
        |(a, b): (MyF64, MyF64)| {
            let (a, b) = (a.0, b.0);
            if a.is_nan() || b.is_nan() {
                return None;
            }

            if a.is_nan() || b.is_nan() {
                Some(-1)
            } else if a < b {
                Some(-1)
            } else if a > b {
                Some(1)
            } else {
                Some(0)
            }
        },
        "builtins::float::cmp::__gedf2(a, b)",
    );
    gen(
        |(a, b): (MyF32, MyF32)| {
            let (a, b) = (a.0, b.0);
            if a.is_nan() || b.is_nan() {
                return None;
            }

            if a.is_nan() || b.is_nan() {
                Some(-1)
            } else if a < b {
                Some(-1)
            } else if a > b {
                Some(1)
            } else {
                Some(0)
            }
        },
        "builtins::float::cmp::__gesf2(a, b)",
    );
    gen(
        |(a, b): (MyF64, MyF64)| {
            let (a, b) = (a.0, b.0);
            if a.is_nan() || b.is_nan() {
                return None;
            }

            if a.is_nan() || b.is_nan() {
                Some(1)
            } else if a < b {
                Some(-1)
            } else if a > b {
                Some(1)
            } else {
                Some(0)
            }
        },
        "builtins::float::cmp::__ledf2(a, b)",
    );
    gen(
        |(a, b): (MyF32, MyF32)| {
            let (a, b) = (a.0, b.0);
            if a.is_nan() || b.is_nan() {
                return None;
            }

            if a.is_nan() || b.is_nan() {
                Some(1)
            } else if a < b {
                Some(-1)
            } else if a > b {
                Some(1)
            } else {
                Some(0)
            }
        },
        "builtins::float::cmp::__lesf2(a, b)",
    );

    gen(
        |(a, b): (MyF32, MyF32)| {
            let c = a.0.is_nan() || b.0.is_nan();
            Some(c as i32)
        },
        "builtins::float::cmp::__unordsf2(a, b)",
    );

    gen(
        |(a, b): (MyF64, MyF64)| {
            let c = a.0.is_nan() || b.0.is_nan();
            Some(c as i32)
        },
        "builtins::float::cmp::__unorddf2(a, b)",
    );

    if target_arch_arm {
        gen(
            |(a, b): (MyF32, MyF32)| {
                if a.0.is_nan() || b.0.is_nan() {
                    return None;
                }
                let c = (a.0 <= b.0) as i32;
                Some(c)
            },
            "builtins::float::cmp::__aeabi_fcmple(a, b)",
        );

        gen(
            |(a, b): (MyF32, MyF32)| {
                if a.0.is_nan() || b.0.is_nan() {
                    return None;
                }
                let c = (a.0 >= b.0) as i32;
                Some(c)
            },
            "builtins::float::cmp::__aeabi_fcmpge(a, b)",
        );

        gen(
            |(a, b): (MyF32, MyF32)| {
                if a.0.is_nan() || b.0.is_nan() {
                    return None;
                }
                let c = (a.0 == b.0) as i32;
                Some(c)
            },
            "builtins::float::cmp::__aeabi_fcmpeq(a, b)",
        );

        gen(
            |(a, b): (MyF32, MyF32)| {
                if a.0.is_nan() || b.0.is_nan() {
                    return None;
                }
                let c = (a.0 < b.0) as i32;
                Some(c)
            },
            "builtins::float::cmp::__aeabi_fcmplt(a, b)",
        );

        gen(
            |(a, b): (MyF32, MyF32)| {
                if a.0.is_nan() || b.0.is_nan() {
                    return None;
                }
                let c = (a.0 > b.0) as i32;
                Some(c)
            },
            "builtins::float::cmp::__aeabi_fcmpgt(a, b)",
        );

        gen(
            |(a, b): (MyF64, MyF64)| {
                if a.0.is_nan() || b.0.is_nan() {
                    return None;
                }
                let c = (a.0 <= b.0) as i32;
                Some(c)
            },
            "builtins::float::cmp::__aeabi_dcmple(a, b)",
        );

        gen(
            |(a, b): (MyF64, MyF64)| {
                if a.0.is_nan() || b.0.is_nan() {
                    return None;
                }
                let c = (a.0 >= b.0) as i32;
                Some(c)
            },
            "builtins::float::cmp::__aeabi_dcmpge(a, b)",
        );

        gen(
            |(a, b): (MyF64, MyF64)| {
                if a.0.is_nan() || b.0.is_nan() {
                    return None;
                }
                let c = (a.0 == b.0) as i32;
                Some(c)
            },
            "builtins::float::cmp::__aeabi_dcmpeq(a, b)",
        );

        gen(
            |(a, b): (MyF64, MyF64)| {
                if a.0.is_nan() || b.0.is_nan() {
                    return None;
                }
                let c = (a.0 < b.0) as i32;
                Some(c)
            },
            "builtins::float::cmp::__aeabi_dcmplt(a, b)",
        );

        gen(
            |(a, b): (MyF64, MyF64)| {
                if a.0.is_nan() || b.0.is_nan() {
                    return None;
                }
                let c = (a.0 > b.0) as i32;
                Some(c)
            },
            "builtins::float::cmp::__aeabi_dcmpgt(a, b)",
        );

        gen(
            |(a, b): (LargeF32, LargeF32)| {
                if a.0.is_nan() || b.0.is_nan() {
                    return None;
                }
                Some((a.0 >= b.0) as i32)
            },
            "builtins::float::cmp::__gesf2vfp(a, b)",
        );
        gen(
            |(a, b): (MyF64, MyF64)| {
                if a.0.is_nan() || b.0.is_nan() {
                    return None;
                }
                Some((a.0 >= b.0) as i32)
            },
            "builtins::float::cmp::__gedf2vfp(a, b)",
        );
        gen(
            |(a, b): (LargeF32, LargeF32)| {
                if a.0.is_nan() || b.0.is_nan() {
                    return None;
                }
                Some((a.0 > b.0) as i32)
            },
            "builtins::float::cmp::__gtsf2vfp(a, b)",
        );
        gen(
            |(a, b): (MyF64, MyF64)| {
                if a.0.is_nan() || b.0.is_nan() {
                    return None;
                }
                Some((a.0 > b.0) as i32)
            },
            "builtins::float::cmp::__gtdf2vfp(a, b)",
        );
        gen(
            |(a, b): (LargeF32, LargeF32)| {
                if a.0.is_nan() || b.0.is_nan() {
                    return None;
                }
                Some((a.0 < b.0) as i32)
            },
            "builtins::float::cmp::__ltsf2vfp(a, b)",
        );
        gen(
            |(a, b): (MyF64, MyF64)| {
                if a.0.is_nan() || b.0.is_nan() {
                    return None;
                }
                Some((a.0 < b.0) as i32)
            },
            "builtins::float::cmp::__ltdf2vfp(a, b)",
        );
        gen(
            |(a, b): (LargeF32, LargeF32)| {
                if a.0.is_nan() || b.0.is_nan() {
                    return None;
                }
                Some((a.0 <= b.0) as i32)
            },
            "builtins::float::cmp::__lesf2vfp(a, b)",
        );
        gen(
            |(a, b): (MyF64, MyF64)| {
                if a.0.is_nan() || b.0.is_nan() {
                    return None;
                }
                Some((a.0 <= b.0) as i32)
            },
            "builtins::float::cmp::__ledf2vfp(a, b)",
        );
        gen(
            |(a, b): (LargeF32, LargeF32)| {
                if a.0.is_nan() || b.0.is_nan() {
                    return None;
                }
                Some((a.0 != b.0) as i32)
            },
            "builtins::float::cmp::__nesf2vfp(a, b)",
        );
        gen(
            |(a, b): (MyF64, MyF64)| {
                if a.0.is_nan() || b.0.is_nan() {
                    return None;
                }
                Some((a.0 != b.0) as i32)
            },
            "builtins::float::cmp::__nedf2vfp(a, b)",
        );
        gen(
            |(a, b): (LargeF32, LargeF32)| {
                if a.0.is_nan() || b.0.is_nan() {
                    return None;
                }
                Some((a.0 == b.0) as i32)
            },
            "builtins::float::cmp::__eqsf2vfp(a, b)",
        );
        gen(
            |(a, b): (MyF64, MyF64)| {
                if a.0.is_nan() || b.0.is_nan() {
                    return None;
                }
                Some((a.0 == b.0) as i32)
            },
            "builtins::float::cmp::__eqdf2vfp(a, b)",
        );
    }

    // float/extend.rs
    gen(
        |a: MyF32| {
            if a.0.is_nan() {
                return None;
            }
            Some(f64::from(a.0))
        },
        "builtins::float::extend::__extendsfdf2(a)",
    );
    if target_arch_arm {
        gen(
            |a: LargeF32| {
                if a.0.is_nan() {
                    return None;
                }
                Some(f64::from(a.0))
            },
            "builtins::float::extend::__extendsfdf2vfp(a)",
        );
    }

    // float/conv.rs
    gen(
        |a: MyF64| i64::cast(a.0),
        "builtins::float::conv::__fixdfdi(a)",
    );
    gen(
        |a: MyF64| i32::cast(a.0),
        "builtins::float::conv::__fixdfsi(a)",
    );
    gen(
        |a: MyF32| i64::cast(a.0),
        "builtins::float::conv::__fixsfdi(a)",
    );
    gen(
        |a: MyF32| i32::cast(a.0),
        "builtins::float::conv::__fixsfsi(a)",
    );
    gen(
        |a: MyF32| i128::cast(a.0),
        "builtins::float::conv::__fixsfti(a)",
    );
    gen(
        |a: MyF64| i128::cast(a.0),
        "builtins::float::conv::__fixdfti(a)",
    );
    gen(
        |a: MyF64| u64::cast(a.0),
        "builtins::float::conv::__fixunsdfdi(a)",
    );
    gen(
        |a: MyF64| u32::cast(a.0),
        "builtins::float::conv::__fixunsdfsi(a)",
    );
    gen(
        |a: MyF32| u64::cast(a.0),
        "builtins::float::conv::__fixunssfdi(a)",
    );
    gen(
        |a: MyF32| u32::cast(a.0),
        "builtins::float::conv::__fixunssfsi(a)",
    );
    gen(
        |a: MyF32| u128::cast(a.0),
        "builtins::float::conv::__fixunssfti(a)",
    );
    gen(
        |a: MyF64| u128::cast(a.0),
        "builtins::float::conv::__fixunsdfti(a)",
    );
    gen(
        |a: MyI64| Some(a.0 as f64),
        "builtins::float::conv::__floatdidf(a)",
    );
    gen(
        |a: MyI32| Some(a.0 as f64),
        "builtins::float::conv::__floatsidf(a)",
    );
    gen(
        |a: MyI32| Some(a.0 as f32),
        "builtins::float::conv::__floatsisf(a)",
    );
    gen(
        |a: MyU64| Some(a.0 as f64),
        "builtins::float::conv::__floatundidf(a)",
    );
    gen(
        |a: MyU32| Some(a.0 as f64),
        "builtins::float::conv::__floatunsidf(a)",
    );
    gen(
        |a: MyU32| Some(a.0 as f32),
        "builtins::float::conv::__floatunsisf(a)",
    );
    gen(
        |a: MyU128| Some(a.0 as f32),
        "builtins::float::conv::__floatuntisf(a)",
    );
    if !target_arch_mips {
        gen(
            |a: MyI128| Some(a.0 as f32),
            "builtins::float::conv::__floattisf(a)",
        );
        gen(
            |a: MyI128| Some(a.0 as f64),
            "builtins::float::conv::__floattidf(a)",
        );
        gen(
            |a: MyU128| Some(a.0 as f64),
            "builtins::float::conv::__floatuntidf(a)",
        );
    }

    // float/pow.rs
    gen(
        |(a, b): (MyF64, MyI32)| {
            let c = a.0.powi(b.0);
            if a.0.is_nan() || c.is_nan() {
                None
            } else {
                Some(c)
            }
        },
        "builtins::float::pow::__powidf2(a, b)",
    );
    gen(
        |(a, b): (MyF32, MyI32)| {
            let c = a.0.powi(b.0);
            if a.0.is_nan() || c.is_nan() {
                None
            } else {
                Some(c)
            }
        },
        "builtins::float::pow::__powisf2(a, b)",
    );

    // float/sub.rs
    gen(
        |(a, b): (MyF64, MyF64)| {
            let c = a.0 - b.0;
            if a.0.is_nan() || b.0.is_nan() || c.is_nan() {
                None
            } else {
                Some(c)
            }
        },
        "builtins::float::sub::__subdf3(a, b)",
    );
    gen(
        |(a, b): (MyF32, MyF32)| {
            let c = a.0 - b.0;
            if a.0.is_nan() || b.0.is_nan() || c.is_nan() {
                None
            } else {
                Some(c)
            }
        },
        "builtins::float::sub::__subsf3(a, b)",
    );

    if target_arch_arm {
        gen(
            |(a, b): (MyF64, MyF64)| {
                let c = a.0 - b.0;
                if a.0.is_nan() || b.0.is_nan() || c.is_nan() {
                    None
                } else {
                    Some(c)
                }
            },
            "builtins::float::sub::__subdf3vfp(a, b)",
        );
        gen(
            |(a, b): (LargeF32, LargeF32)| {
                let c = a.0 - b.0;
                if a.0.is_nan() || b.0.is_nan() || c.is_nan() {
                    None
                } else {
                    Some(c)
                }
            },
            "builtins::float::sub::__subsf3vfp(a, b)",
        );
    }

    // float/mul.rs
    gen(
        |(a, b): (MyF64, MyF64)| {
            let c = a.0 * b.0;
            if a.0.is_nan() || b.0.is_nan() || c.is_nan() {
                None
            } else {
                Some(c)
            }
        },
        "builtins::float::mul::__muldf3(a, b)",
    );
    gen(
        |(a, b): (LargeF32, LargeF32)| {
            let c = a.0 * b.0;
            if a.0.is_nan() || b.0.is_nan() || c.is_nan() {
                None
            } else {
                Some(c)
            }
        },
        "builtins::float::mul::__mulsf3(a, b)",
    );

    if target_arch_arm {
        gen(
            |(a, b): (MyF64, MyF64)| {
                let c = a.0 * b.0;
                if a.0.is_nan() || b.0.is_nan() || c.is_nan() {
                    None
                } else {
                    Some(c)
                }
            },
            "builtins::float::mul::__muldf3vfp(a, b)",
        );
        gen(
            |(a, b): (LargeF32, LargeF32)| {
                let c = a.0 * b.0;
                if a.0.is_nan() || b.0.is_nan() || c.is_nan() {
                    None
                } else {
                    Some(c)
                }
            },
            "builtins::float::mul::__mulsf3vfp(a, b)",
        );
    }

    // float/div.rs
    gen(
        |(a, b): (MyF64, MyF64)| {
            if b.0 == 0.0 {
                return None;
            }
            let c = a.0 / b.0;
            if a.0.is_nan()
                || b.0.is_nan()
                || c.is_nan()
                || c.abs() <= unsafe { mem::transmute(4503599627370495u64) }
            {
                None
            } else {
                Some(c)
            }
        },
        "builtins::float::div::__divdf3(a, b)",
    );
    gen(
        |(a, b): (LargeF32, LargeF32)| {
            if b.0 == 0.0 {
                return None;
            }
            let c = a.0 / b.0;
            if a.0.is_nan()
                || b.0.is_nan()
                || c.is_nan()
                || c.abs() <= unsafe { mem::transmute(16777215u32) }
            {
                None
            } else {
                Some(c)
            }
        },
        "builtins::float::div::__divsf3(a, b)",
    );

    if target_arch_arm {
        gen(
            |(a, b): (MyF64, MyF64)| {
                if b.0 == 0.0 {
                    return None;
                }
                let c = a.0 / b.0;
                if a.0.is_nan()
                    || b.0.is_nan()
                    || c.is_nan()
                    || c.abs() <= unsafe { mem::transmute(4503599627370495u64) }
                {
                    None
                } else {
                    Some(c)
                }
            },
            "builtins::float::div::__divdf3vfp(a, b)",
        );
        gen(
            |(a, b): (LargeF32, LargeF32)| {
                if b.0 == 0.0 {
                    return None;
                }
                let c = a.0 / b.0;
                if a.0.is_nan()
                    || b.0.is_nan()
                    || c.is_nan()
                    || c.abs() <= unsafe { mem::transmute(16777215u32) }
                {
                    None
                } else {
                    Some(c)
                }
            },
            "builtins::float::div::__divsf3vfp(a, b)",
        );
    }

    // int/addsub.rs
    gen(
        |(a, b): (MyU128, MyU128)| Some(a.0.wrapping_add(b.0)),
        "builtins::int::addsub::__rust_u128_add(a, b)",
    );
    gen(
        |(a, b): (MyI128, MyI128)| Some(a.0.wrapping_add(b.0)),
        "builtins::int::addsub::__rust_i128_add(a, b)",
    );
    gen(
        |(a, b): (MyU128, MyU128)| Some(a.0.overflowing_add(b.0)),
        "builtins::int::addsub::__rust_u128_addo(a, b)",
    );
    gen(
        |(a, b): (MyI128, MyI128)| Some(a.0.overflowing_add(b.0)),
        "builtins::int::addsub::__rust_i128_addo(a, b)",
    );
    gen(
        |(a, b): (MyU128, MyU128)| Some(a.0.wrapping_sub(b.0)),
        "builtins::int::addsub::__rust_u128_sub(a, b)",
    );
    gen(
        |(a, b): (MyI128, MyI128)| Some(a.0.wrapping_sub(b.0)),
        "builtins::int::addsub::__rust_i128_sub(a, b)",
    );
    gen(
        |(a, b): (MyU128, MyU128)| Some(a.0.overflowing_sub(b.0)),
        "builtins::int::addsub::__rust_u128_subo(a, b)",
    );
    gen(
        |(a, b): (MyI128, MyI128)| Some(a.0.overflowing_sub(b.0)),
        "builtins::int::addsub::__rust_i128_subo(a, b)",
    );

    // int/mul.rs
    gen(
        |(a, b): (MyU64, MyU64)| Some(a.0.wrapping_mul(b.0)),
        "builtins::int::mul::__muldi3(a, b)",
    );
    gen(
        |(a, b): (MyI64, MyI64)| Some(a.0.overflowing_mul(b.0)),
        "{
            let mut o = 2;
            let c = builtins::int::mul::__mulodi4(a, b, &mut o);
            (c, match o { 0 => false, 1 => true, _ => panic!() })
        }",
    );
    gen(
        |(a, b): (MyI32, MyI32)| Some(a.0.overflowing_mul(b.0)),
        "{
            let mut o = 2;
            let c = builtins::int::mul::__mulosi4(a, b, &mut o);
            (c, match o { 0 => false, 1 => true, _ => panic!() })
        }",
    );
    gen(
        |(a, b): (MyI128, MyI128)| Some(a.0.wrapping_mul(b.0)),
        "builtins::int::mul::__multi3(a, b)",
    );
    gen(
        |(a, b): (MyI128, MyI128)| Some(a.0.overflowing_mul(b.0)),
        "{
            let mut o = 2;
            let c = builtins::int::mul::__muloti4(a, b, &mut o);
            (c, match o { 0 => false, 1 => true, _ => panic!() })
        }",
    );

    // int/sdiv.rs
    gen(
        |(a, b): (MyI64, MyI64)| {
            if b.0 == 0 {
                None
            } else {
                Some(a.0 / b.0)
            }
        },
        "builtins::int::sdiv::__divdi3(a, b)",
    );
    gen(
        |(a, b): (MyI64, MyI64)| {
            if b.0 == 0 {
                None
            } else {
                Some((a.0 / b.0, a.0 % b.0))
            }
        },
        "{
            let mut r = 0;
            (builtins::int::sdiv::__divmoddi4(a, b, &mut r), r)
        }",
    );
    gen(
        |(a, b): (MyI32, MyI32)| {
            if b.0 == 0 {
                None
            } else {
                Some((a.0 / b.0, a.0 % b.0))
            }
        },
        "{
            let mut r = 0;
            (builtins::int::sdiv::__divmodsi4(a, b, &mut r), r)
        }",
    );
    gen(
        |(a, b): (MyI32, MyI32)| {
            if b.0 == 0 {
                None
            } else {
                Some(a.0 / b.0)
            }
        },
        "builtins::int::sdiv::__divsi3(a, b)",
    );
    gen(
        |(a, b): (MyI32, MyI32)| {
            if b.0 == 0 {
                None
            } else {
                Some(a.0 % b.0)
            }
        },
        "builtins::int::sdiv::__modsi3(a, b)",
    );
    gen(
        |(a, b): (MyI64, MyI64)| {
            if b.0 == 0 {
                None
            } else {
                Some(a.0 % b.0)
            }
        },
        "builtins::int::sdiv::__moddi3(a, b)",
    );
    gen(
        |(a, b): (MyI128, MyI128)| {
            if b.0 == 0 {
                None
            } else {
                Some(a.0 / b.0)
            }
        },
        "builtins::int::sdiv::__divti3(a, b)",
    );
    gen(
        |(a, b): (MyI128, MyI128)| {
            if b.0 == 0 {
                None
            } else {
                Some(a.0 % b.0)
            }
        },
        "builtins::int::sdiv::__modti3(a, b)",
    );

    // int/shift.rs
    gen(
        |(a, b): (MyU32, MyU32)| Some(a.0 << (b.0 % 32)),
        "builtins::int::shift::__ashlsi3(a, b % 32)",
    );
    gen(
        |(a, b): (MyU64, MyU32)| Some(a.0 << (b.0 % 64)),
        "builtins::int::shift::__ashldi3(a, b % 64)",
    );
    gen(
        |(a, b): (MyU128, MyU32)| Some(a.0 << (b.0 % 128)),
        "builtins::int::shift::__ashlti3(a, b % 128)",
    );
    gen(
        |(a, b): (MyI32, MyU32)| Some(a.0 >> (b.0 % 32)),
        "builtins::int::shift::__ashrsi3(a, b % 32)",
    );
    gen(
        |(a, b): (MyI64, MyU32)| Some(a.0 >> (b.0 % 64)),
        "builtins::int::shift::__ashrdi3(a, b % 64)",
    );
    gen(
        |(a, b): (MyI128, MyU32)| Some(a.0 >> (b.0 % 128)),
        "builtins::int::shift::__ashrti3(a, b % 128)",
    );
    gen(
        |(a, b): (MyU32, MyU32)| Some(a.0 >> (b.0 % 32)),
        "builtins::int::shift::__lshrsi3(a, b % 32)",
    );
    gen(
        |(a, b): (MyU64, MyU32)| Some(a.0 >> (b.0 % 64)),
        "builtins::int::shift::__lshrdi3(a, b % 64)",
    );
    gen(
        |(a, b): (MyU128, MyU32)| Some(a.0 >> (b.0 % 128)),
        "builtins::int::shift::__lshrti3(a, b % 128)",
    );

    // int/udiv.rs
    gen(
        |(a, b): (MyU64, MyU64)| {
            if b.0 == 0 {
                None
            } else {
                Some(a.0 / b.0)
            }
        },
        "builtins::int::udiv::__udivdi3(a, b)",
    );
    gen(
        |(a, b): (MyU64, MyU64)| {
            if b.0 == 0 {
                None
            } else {
                Some((a.0 / b.0, a.0 % b.0))
            }
        },
        "{
            let mut r = 0;
            (builtins::int::udiv::__udivmoddi4(a, b, Some(&mut r)), r)
        }",
    );
    gen(
        |(a, b): (MyU32, MyU32)| {
            if b.0 == 0 {
                None
            } else {
                Some((a.0 / b.0, a.0 % b.0))
            }
        },
        "{
            let mut r = 0;
            (builtins::int::udiv::__udivmodsi4(a, b, Some(&mut r)), r)
        }",
    );
    gen(
        |(a, b): (MyU32, MyU32)| {
            if b.0 == 0 {
                None
            } else {
                Some(a.0 / b.0)
            }
        },
        "builtins::int::udiv::__udivsi3(a, b)",
    );
    gen(
        |(a, b): (MyU32, MyU32)| {
            if b.0 == 0 {
                None
            } else {
                Some(a.0 % b.0)
            }
        },
        "builtins::int::udiv::__umodsi3(a, b)",
    );
    gen(
        |(a, b): (MyU64, MyU64)| {
            if b.0 == 0 {
                None
            } else {
                Some(a.0 % b.0)
            }
        },
        "builtins::int::udiv::__umoddi3(a, b)",
    );
    gen(
        |(a, b): (MyU128, MyU128)| {
            if b.0 == 0 {
                None
            } else {
                Some(a.0 / b.0)
            }
        },
        "builtins::int::udiv::__udivti3(a, b)",
    );
    gen(
        |(a, b): (MyU128, MyU128)| {
            if b.0 == 0 {
                None
            } else {
                Some(a.0 % b.0)
            }
        },
        "builtins::int::udiv::__umodti3(a, b)",
    );
    gen(
        |(a, b): (MyU128, MyU128)| {
            if b.0 == 0 {
                None
            } else {
                Some((a.0 / b.0, a.0 % b.0))
            }
        },
        "{
            let mut r = 0;
            (builtins::int::udiv::__udivmodti4(a, b, Some(&mut r)), r)
        }",
    );
}

macro_rules! gen_float {
    ($name:ident,
     $fty:ident,
     $uty:ident,
     $bits:expr,
     $significand_bits:expr) => {
        pub fn $name<R>(rng: &mut R) -> $fty
        where
            R: Rng + ?Sized,
        {
            const BITS: u8 = $bits;
            const SIGNIFICAND_BITS: u8 = $significand_bits;

            const SIGNIFICAND_MASK: $uty = (1 << SIGNIFICAND_BITS) - 1;
            const SIGN_MASK: $uty = (1 << (BITS - 1));
            const EXPONENT_MASK: $uty = !(SIGN_MASK | SIGNIFICAND_MASK);

            fn mk_f32(sign: bool, exponent: $uty, significand: $uty) -> $fty {
                unsafe {
                    mem::transmute(
                        ((sign as $uty) << (BITS - 1))
                            | ((exponent & EXPONENT_MASK) << SIGNIFICAND_BITS)
                            | (significand & SIGNIFICAND_MASK),
                    )
                }
            }

            if rng.gen_range(0, 10) == 1 {
                // Special values
                *[
                    -0.0,
                    0.0,
                    ::std::$fty::MIN,
                    ::std::$fty::MIN_POSITIVE,
                    ::std::$fty::MAX,
                    ::std::$fty::NAN,
                    ::std::$fty::INFINITY,
                    -::std::$fty::INFINITY,
                ]
                .choose(rng)
                .unwrap()
            } else if rng.gen_range(0, 10) == 1 {
                // NaN patterns
                mk_f32(rng.gen(), rng.gen(), 0)
            } else if rng.gen() {
                // Denormalized
                mk_f32(rng.gen(), 0, rng.gen())
            } else {
                // Random anything
                mk_f32(rng.gen(), rng.gen(), rng.gen())
            }
        }
    };
}

gen_float!(gen_f32, f32, u32, 32, 23);
gen_float!(gen_f64, f64, u64, 64, 52);

macro_rules! gen_large_float {
    ($name:ident,
     $fty:ident,
     $uty:ident,
     $bits:expr,
     $significand_bits:expr) => {
        pub fn $name<R>(rng: &mut R) -> $fty
        where
            R: Rng + ?Sized,
        {
            const BITS: u8 = $bits;
            const SIGNIFICAND_BITS: u8 = $significand_bits;

            const SIGNIFICAND_MASK: $uty = (1 << SIGNIFICAND_BITS) - 1;
            const SIGN_MASK: $uty = (1 << (BITS - 1));
            const EXPONENT_MASK: $uty = !(SIGN_MASK | SIGNIFICAND_MASK);

            fn mk_f32(sign: bool, exponent: $uty, significand: $uty) -> $fty {
                unsafe {
                    mem::transmute(
                        ((sign as $uty) << (BITS - 1))
                            | ((exponent & EXPONENT_MASK) << SIGNIFICAND_BITS)
                            | (significand & SIGNIFICAND_MASK),
                    )
                }
            }

            if rng.gen_range(0, 10) == 1 {
                // Special values
                *[
                    -0.0,
                    0.0,
                    ::std::$fty::MIN,
                    ::std::$fty::MIN_POSITIVE,
                    ::std::$fty::MAX,
                    ::std::$fty::NAN,
                    ::std::$fty::INFINITY,
                    -::std::$fty::INFINITY,
                ]
                .choose(rng)
                .unwrap()
            } else if rng.gen_range(0, 10) == 1 {
                // NaN patterns
                mk_f32(rng.gen(), rng.gen(), 0)
            } else if rng.gen() {
                // Denormalized
                mk_f32(rng.gen(), 0, rng.gen())
            } else {
                // Random anything
                rng.gen::<$fty>()
            }
        }
    };
}

gen_large_float!(gen_large_f32, f32, u32, 32, 23);
gen_large_float!(gen_large_f64, f64, u64, 64, 52);

trait TestInput: Hash + Eq + fmt::Debug {
    fn ty_name() -> String;
    fn generate_lets(container: &str, cnt: &mut u8) -> String;
    fn generate_static(&self, dst: &mut String);
}

trait TestOutput {
    fn ty_name() -> String;
    fn generate_static(&self, dst: &mut String);
    fn generate_expr(container: &str) -> String;
}

fn gen<F, A, R>(mut generate: F, test: &str)
where
    F: FnMut(A) -> Option<R>,
    A: TestInput + Copy,
    R: TestOutput,
    rand::distributions::Standard: rand::distributions::Distribution<A>,
{
    let rng = &mut rand::thread_rng();
    let testname = test.split("::").last().unwrap().split("(").next().unwrap();
    let out_dir = PathBuf::from(env::var_os("OUT_DIR").unwrap());
    let out_file = out_dir.join("generated.rs");

    let mut testcases = HashMap::new();
    let mut n = NTESTS;
    while n > 0 {
        let input: A = rng.gen();
        if testcases.contains_key(&input) {
            continue;
        }
        let output = match generate(input) {
            Some(o) => o,
            None => continue,
        };
        testcases.insert(input, output);
        n -= 1;
    }

    let mut contents = String::new();
    contents.push_str(&format!("mod {} {{\nuse super::*;\n", testname));
    contents.push_str("#[test]\n");
    contents.push_str("fn test() {\n");
    contents.push_str(&format!(
        "static TESTS: [({}, {}); {}] = [\n",
        A::ty_name(),
        R::ty_name(),
        NTESTS
    ));
    for (input, output) in testcases {
        contents.push_str("    (");
        input.generate_static(&mut contents);
        contents.push_str(", ");
        output.generate_static(&mut contents);
        contents.push_str("),\n");
    }
    contents.push_str("];\n");

    contents.push_str(&format!(
        r#"
        for &(inputs, output) in TESTS.iter() {{
            {}
            assert_eq!({}, {}, "inputs {{:?}}", inputs)
        }}
    "#,
        A::generate_lets("inputs", &mut 0),
        R::generate_expr("output"),
        test,
    ));
    contents.push_str("\n}\n");
    contents.push_str("\n}\n");

    OpenOptions::new()
        .write(true)
        .append(true)
        .create(true)
        .open(out_file)
        .unwrap()
        .write_all(contents.as_bytes())
        .unwrap();
}

macro_rules! my_float {
    ($(struct $name:ident($inner:ident) = $gen:ident;)*) => ($(
        #[derive(Debug, Clone, Copy)]
        struct $name($inner);

        impl TestInput for $name {
            fn ty_name() -> String {
                format!("u{}", &stringify!($inner)[1..])
            }

            fn generate_lets(container: &str, cnt: &mut u8) -> String {
                let me = *cnt;
                *cnt += 1;
                format!("let {} = {}::from_bits({});\n",
                        (b'a' + me) as char,
                        stringify!($inner),
                        container)
            }

            fn generate_static(&self, dst: &mut String) {
                write!(dst, "{}", self.0.to_bits()).unwrap();
            }
        }

        impl rand::distributions::Distribution<$name> for rand::distributions::Standard {
            fn sample<R: rand::Rng + ?Sized >(&self, r: &mut R) -> $name {
                $name($gen(r))
            }
        }

        impl Hash for $name {
            fn hash<H: Hasher>(&self, h: &mut H) {
                self.0.to_bits().hash(h)
            }
        }

        impl PartialEq for $name {
            fn eq(&self, other: &$name) -> bool {
                self.0.to_bits() == other.0.to_bits()
            }
        }

        impl Eq for $name {}

    )*)
}

my_float! {
    struct MyF64(f64) = gen_f64;
    struct LargeF64(f64) = gen_large_f64;
    struct MyF32(f32) = gen_f32;
    struct LargeF32(f32) = gen_large_f32;
}

macro_rules! my_integer {
    ($(struct $name:ident($inner:ident);)*) => ($(
        #[derive(Debug, Clone, Copy, Hash, PartialEq, Eq)]
        struct $name($inner);

        impl TestInput for $name {
            fn ty_name() -> String {
                stringify!($inner).to_string()
            }

            fn generate_lets(container: &str, cnt: &mut u8) -> String {
                let me = *cnt;
                *cnt += 1;
                format!("let {} = {};\n",
                        (b'a' + me) as char,
                        container)
            }

            fn generate_static(&self, dst: &mut String) {
                write!(dst, "{}", self.0).unwrap();
            }
        }

        impl rand::distributions::Distribution<$name> for rand::distributions::Standard {
            fn sample<R: rand::Rng + ?Sized >(&self, r: &mut R) -> $name {
                let bits = (0 as $inner).count_zeros();
                let mut mk = || {
                    if r.gen_range(0, 10) == 1 {
                        *[
                            ::std::$inner::MAX >> (bits / 2),
                            0,
                            ::std::$inner::MIN >> (bits / 2),
                        ].choose(r).unwrap()
                    } else {
                        r.gen::<$inner>()
                    }
                };
                let a = mk();
                let b = mk();
                $name((a << (bits / 2)) | (b & (!0 << (bits / 2))))
            }
        }
    )*)
}

my_integer! {
    struct MyI32(i32);
    struct MyI64(i64);
    struct MyI128(i128);
    struct MyU16(u16);
    struct MyU32(u32);
    struct MyU64(u64);
    struct MyU128(u128);
}

impl<A, B> TestInput for (A, B)
where
    A: TestInput,
    B: TestInput,
{
    fn ty_name() -> String {
        format!("({}, {})", A::ty_name(), B::ty_name())
    }

    fn generate_lets(container: &str, cnt: &mut u8) -> String {
        format!(
            "{}{}",
            A::generate_lets(&format!("{}.0", container), cnt),
            B::generate_lets(&format!("{}.1", container), cnt)
        )
    }

    fn generate_static(&self, dst: &mut String) {
        dst.push_str("(");
        self.0.generate_static(dst);
        dst.push_str(", ");
        self.1.generate_static(dst);
        dst.push_str(")");
    }
}

impl TestOutput for f64 {
    fn ty_name() -> String {
        "u64".to_string()
    }

    fn generate_static(&self, dst: &mut String) {
        write!(dst, "{}", self.to_bits()).unwrap();
    }

    fn generate_expr(container: &str) -> String {
        format!("f64::from_bits({})", container)
    }
}

impl TestOutput for f32 {
    fn ty_name() -> String {
        "u32".to_string()
    }

    fn generate_static(&self, dst: &mut String) {
        write!(dst, "{}", self.to_bits()).unwrap();
    }

    fn generate_expr(container: &str) -> String {
        format!("f32::from_bits({})", container)
    }
}

macro_rules! plain_test_output {
    ($($i:tt)*) => ($(
        impl TestOutput for $i {
            fn ty_name() -> String {
                stringify!($i).to_string()
            }

            fn generate_static(&self, dst: &mut String) {
                write!(dst, "{}", self).unwrap();
            }

            fn generate_expr(container: &str) -> String {
                container.to_string()
            }
        }
    )*)
}

plain_test_output!(i32 i64 i128 u32 u64 u128 bool);

impl<A, B> TestOutput for (A, B)
where
    A: TestOutput,
    B: TestOutput,
{
    fn ty_name() -> String {
        format!("({}, {})", A::ty_name(), B::ty_name())
    }

    fn generate_static(&self, dst: &mut String) {
        dst.push_str("(");
        self.0.generate_static(dst);
        dst.push_str(", ");
        self.1.generate_static(dst);
        dst.push_str(")");
    }

    fn generate_expr(container: &str) -> String {
        container.to_string()
    }
}

trait FromFloat<T>: Sized {
    fn cast(src: T) -> Option<Self>;
}

macro_rules! from_float {
    ($($src:ident => $($dst:ident),+);+;) => {
        $(
            $(
                impl FromFloat<$src> for $dst {
                    fn cast(src: $src) -> Option<$dst> {
                        use std::{$dst, $src};

                        if src.is_nan() ||
                            src.is_infinite() ||
                            src < std::$dst::MIN as $src ||
                            src > std::$dst::MAX as $src
                        {
                            None
                        } else {
                            Some(src as $dst)
                        }
                    }
                }
            )+
        )+
    }
}

from_float! {
    f32 => i32, i64, i128, u32, u64, u128;
    f64 => i32, i64, i128, u32, u64, u128;
}
