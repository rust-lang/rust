// Basic test for free regions in the NLL code. This test does not
// report an error because of the (implied) bound that `'b: 'a`.

// check-pass
// compile-flags:-Zborrowck=mir -Zverbose

fn foo<'a, 'b>(x: &'a &'b u32) -> &'a u32 {
    &**x
}

fn main() {}
