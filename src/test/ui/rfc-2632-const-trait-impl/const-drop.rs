// run-pass
#![feature(const_trait_impl)]
#![feature(const_fn_trait_bound)]
#![feature(const_mut_refs)]
#![feature(const_panic)]

struct S<'a>(&'a mut u8);

impl<'a> const Drop for S<'a> {
    fn drop(&mut self) {
        *self.0 += 1;
    }
}

const fn a<T: ~const Drop>(_: T) {}

const fn b() -> u8 {
    let mut c = 0;
    let _ = S(&mut c);
    a(S(&mut c));
    c
}

const C: u8 = b();

fn main() {
    assert_eq!(C, 2);
}
