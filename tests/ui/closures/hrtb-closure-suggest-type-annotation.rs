//@ run-rustfix
// When a closure is passed where a higher-ranked `FnOnce` is expected, but the closure
// parameter has no explicit type annotation, the compiler should suggest adding one.
// See <https://github.com/rust-lang/rust/issues/141461>.

fn inner<F>(buf: &mut [u8], func: F) -> usize
where
    F: FnOnce(&mut [u8]) -> Result<usize, isize>,
{
    match func(buf) {
        Ok(n) => n,
        Err(e) => e as usize,
    }
}

fn outer<F>(buf: &mut [u8], func: F) -> usize
where
    F: FnOnce(&mut [u8]) -> Result<usize, isize>,
{
    let outer_closure = |buf| {
        match func(buf) {
            Ok(n) => {
                println!("OK: {n}");
                Ok(n)
            }
            Err(e) => {
                println!("ERR: {e}");
                Err(e)
            }
        }
    };

    inner(buf, outer_closure)
    //~^ ERROR implementation of `FnOnce` is not general enough
    //~| ERROR implementation of `FnOnce` is not general enough
}

fn main() {
    let mut x = [0u8; 16];
    let res = outer(&mut x, |buf| {
        if buf.len() % 2 == 0 { Ok(buf.len()) } else { Err(buf.len() as isize) }
    });
    println!("{res:?}");
}
