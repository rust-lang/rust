// Regression test: issue had to do with "givens" in region inference,
// which were not being considered during the contraction phase.

//@ run-fail
//@ error-pattern:explicit panic
//@ needs-subprocess

struct Parser<'i: 't, 't>(&'i u8, &'t u8);

impl<'i, 't> Parser<'i, 't> {
    fn parse_nested_block<F, T>(&mut self, parse: F) -> Result<T, ()>
        where for<'tt> F: FnOnce(&mut Parser<'i, 'tt>) -> T
    {
        panic!()
    }

    fn expect_exhausted(&mut self) -> Result<(), ()> {
        Ok(())
    }
}

fn main() {
    let x = 0u8;
    Parser(&x, &x).parse_nested_block(|input| input.expect_exhausted()).unwrap();
}
