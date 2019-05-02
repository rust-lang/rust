extern crate shared;

use std::error::Error;
use std::fs::{self, File};
use std::io::Write;
use std::process::{Command, Stdio};

#[macro_use]
mod macros;

fn main() -> Result<(), Box<Error>> {
    const F32: &[u8] = include_bytes!("../../bin/input/f32");

    f32! {
        asinf,
        cbrtf,
        cosf,
        exp2f,
        sinf,
        tanf,
    }

    f32f32! {
        hypotf,
    }

    f32f32f32! {
        fmaf,
    }

    Ok(())
}
