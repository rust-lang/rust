use crate::arch::x86_64::{_rdrand16_step, _rdrand32_step, _rdrand64_step};
use crate::io::BorrowedCursor;

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

pub fn fill_buf(mut cursor: BorrowedCursor<'_, u8>) {
    while cursor.capacity() >= 8 {
        cursor.append(&rdrand64().to_ne_bytes());
    }
    if cursor.capacity() >= 4 {
        cursor.append(&rdrand32().to_ne_bytes());
    }
    if cursor.capacity() >= 2 {
        cursor.append(&rdrand16().to_ne_bytes());
    }
    if cursor.capacity() == 1 {
        cursor.append(&[rdrand16() as u8]);
    }
}
