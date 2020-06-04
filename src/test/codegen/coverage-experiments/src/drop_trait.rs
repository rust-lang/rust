#[inline(always)]
pub fn __incr_cov<T>(_region_loc: &str, result: T) -> T {
    result
}

struct Firework {
    _strength: i32,
}

impl Drop for Firework {
    fn drop(&mut self) {
        __incr_cov("start of drop()", ());
    }
}

fn main() -> Result<(),u8> {
    let _firecracker = Firework { _strength: 1 };

    if __incr_cov("start of main()", true) {
        return __incr_cov("if true", { let _t = Err(1); _t });
    }

    let _tnt = Firework { _strength: 100 };
    Ok(())
}