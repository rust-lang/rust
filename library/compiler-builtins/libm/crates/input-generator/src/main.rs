extern crate rand;

use std::collections::BTreeSet;
use std::error::Error;
use std::fs::{self, File};
use std::io::Write;

use rand::{RngCore, SeedableRng, XorShiftRng};

const NTESTS: usize = 10_000;

fn main() -> Result<(), Box<Error>> {
    let mut rng = XorShiftRng::from_rng(&mut rand::thread_rng())?;

    fs::remove_dir_all("bin").ok();
    fs::create_dir_all("bin/input")?;
    fs::create_dir_all("bin/output")?;

    f32(&mut rng)?;
    f32f32(&mut rng)?;
    f32f32f32(&mut rng)?;
    f32i16(&mut rng)?;
    f64(&mut rng)?;
    f64f64(&mut rng)?;
    f64f64f64(&mut rng)?;
    f64i16(&mut rng)?;

    Ok(())
}

fn f32(rng: &mut XorShiftRng) -> Result<(), Box<Error>> {
    let mut set = BTreeSet::new();

    while set.len() < NTESTS {
        let f = f32::from_bits(rng.next_u32());

        if f.is_nan() {
            continue;
        }

        set.insert(f.to_bits());
    }

    let mut f = File::create("bin/input/f32")?;
    for i in set {
        f.write_all(&i.to_le_bytes())?;
    }

    Ok(())
}

fn f32f32(rng: &mut XorShiftRng) -> Result<(), Box<Error>> {
    let mut f = File::create("bin/input/f32f32")?;
    let mut i = 0;
    while i < NTESTS {
        let x0 = f32::from_bits(rng.next_u32());
        let x1 = f32::from_bits(rng.next_u32());

        if x0.is_nan() || x1.is_nan() {
            continue;
        }

        i += 1;
        f.write_all(&x0.to_bits().to_le_bytes())?;
        f.write_all(&x1.to_bits().to_le_bytes())?;
    }

    Ok(())
}

fn f32i16(rng: &mut XorShiftRng) -> Result<(), Box<Error>> {
    let mut f = File::create("bin/input/f32i16")?;
    let mut i = 0;
    while i < NTESTS {
        let x0 = f32::from_bits(rng.next_u32());
        let x1 = rng.next_u32() as i16;

        if x0.is_nan() {
            continue;
        }

        i += 1;
        f.write_all(&x0.to_bits().to_le_bytes())?;
        f.write_all(&x1.to_le_bytes())?;
    }

    Ok(())
}

fn f32f32f32(rng: &mut XorShiftRng) -> Result<(), Box<Error>> {
    let mut f = File::create("bin/input/f32f32f32")?;
    let mut i = 0;
    while i < NTESTS {
        let x0 = f32::from_bits(rng.next_u32());
        let x1 = f32::from_bits(rng.next_u32());
        let x2 = f32::from_bits(rng.next_u32());

        if x0.is_nan() || x1.is_nan() || x2.is_nan() {
            continue;
        }

        i += 1;
        f.write_all(&x0.to_bits().to_le_bytes())?;
        f.write_all(&x1.to_bits().to_le_bytes())?;
        f.write_all(&x2.to_bits().to_le_bytes())?;
    }

    Ok(())
}

fn f64(rng: &mut XorShiftRng) -> Result<(), Box<Error>> {
    let mut set = BTreeSet::new();

    while set.len() < NTESTS {
        let f = f64::from_bits(rng.next_u64());

        if f.is_nan() {
            continue;
        }

        set.insert(f.to_bits());
    }

    let mut f = File::create("bin/input/f64")?;
    for i in set {
        f.write_all(&i.to_le_bytes())?;
    }

    Ok(())
}

fn f64f64(rng: &mut XorShiftRng) -> Result<(), Box<Error>> {
    let mut f = File::create("bin/input/f64f64")?;
    let mut i = 0;
    while i < NTESTS {
        let x0 = f64::from_bits(rng.next_u64());
        let x1 = f64::from_bits(rng.next_u64());

        if x0.is_nan() || x1.is_nan() {
            continue;
        }

        i += 1;
        f.write_all(&x0.to_bits().to_le_bytes())?;
        f.write_all(&x1.to_bits().to_le_bytes())?;
    }

    Ok(())
}

fn f64f64f64(rng: &mut XorShiftRng) -> Result<(), Box<Error>> {
    let mut f = File::create("bin/input/f64f64f64")?;
    let mut i = 0;
    while i < NTESTS {
        let x0 = f64::from_bits(rng.next_u64());
        let x1 = f64::from_bits(rng.next_u64());
        let x2 = f64::from_bits(rng.next_u64());

        if x0.is_nan() || x1.is_nan() || x2.is_nan() {
            continue;
        }

        i += 1;
        f.write_all(&x0.to_bits().to_le_bytes())?;
        f.write_all(&x1.to_bits().to_le_bytes())?;
        f.write_all(&x2.to_bits().to_le_bytes())?;
    }

    Ok(())
}

fn f64i16(rng: &mut XorShiftRng) -> Result<(), Box<Error>> {
    let mut f = File::create("bin/input/f64i16")?;
    let mut i = 0;
    while i < NTESTS {
        let x0 = f64::from_bits(rng.next_u64());
        let x1 = rng.next_u32() as i16;

        if x0.is_nan() {
            continue;
        }

        i += 1;
        f.write_all(&x0.to_bits().to_le_bytes())?;
        f.write_all(&x1.to_le_bytes())?;
    }

    Ok(())
}
