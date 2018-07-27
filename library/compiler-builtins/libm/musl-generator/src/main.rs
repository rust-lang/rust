extern crate libm;
extern crate shared;

use std::error::Error;
use std::fs::File;
use std::io::Write;

#[macro_use]
mod macros;

fn main() -> Result<(), Box<Error>> {
    f32! {
        acosf,
        asinf,
        atanf,
        cbrtf,
        ceilf,
        cosf,
        coshf,
        exp2f,
        expf,
        expm1f,
        fabsf,
        floorf,
        log10f,
        log1pf,
        log2f,
        logf,
        roundf,
        sinf,
        sinhf,
        sqrtf,
        tanf,
        tanhf,
        truncf,
    }

    f32f32! {
        atan2f,
        fdimf,
        fmodf,
        hypotf,
        powf,
    }

    f32i32! {
        scalbnf,
    }

    f32f32f32! {
        fmaf,
    }

    f64! {
        acos,
        asin,
        atan,
        cbrt,
        ceil,
        cos,
        cosh,
        exp,
        exp2,
        expm1,
        fabs,
        floor,
        log,
        log10,
        log1p,
        log2,
        round,
        sin,
        sinh,
        sqrt,
        tan,
        tanh,
        trunc,
    }

    f64f64! {
        atan2,
        fdim,
        fmod,
        hypot,
        pow,
    }

    f64i32! {
        scalbn,
    }

    f64f64f64! {
        fma,
    }

    Ok(())
}
