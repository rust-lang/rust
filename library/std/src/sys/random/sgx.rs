use crate::arch::x86_64::{_rdrand16_step, _rdrand32_step, _rdrand64_step};

const RETRIES: u32 = 10;

fn fail() -> ! {
    panic!("failed to generate random data");
}

fn rdrand64() -> u64 {
    unsafe {
        let mut ret: u64 = 0;
        for _ in 0..RETRIES {
            if _rdrand64_step(&mut ret) == 1 {
                return ret;
            }
        }

        fail();
    }
}

fn rdrand32() -> u32 {
    unsafe {
        let mut ret: u32 = 0;
        for _ in 0..RETRIES {
            if _rdrand32_step(&mut ret) == 1 {
                return ret;
            }
        }

        fail();
    }
}

fn rdrand16() -> u16 {
    unsafe {
        let mut ret: u16 = 0;
        for _ in 0..RETRIES {
            if _rdrand16_step(&mut ret) == 1 {
                return ret;
            }
        }

        fail();
    }
}

pub fn fill_bytes(bytes: &mut [u8]) {
    let mut chunks = bytes.array_chunks_mut();
    for chunk in &mut chunks {
        *chunk = rdrand64().to_ne_bytes();
    }

    let mut chunks = chunks.into_remainder().array_chunks_mut();
    for chunk in &mut chunks {
        *chunk = rdrand32().to_ne_bytes();
    }

    let mut chunks = chunks.into_remainder().array_chunks_mut();
    for chunk in &mut chunks {
        *chunk = rdrand16().to_ne_bytes();
    }

    if let [byte] = chunks.into_remainder() {
        *byte = rdrand16() as u8;
    }
}
