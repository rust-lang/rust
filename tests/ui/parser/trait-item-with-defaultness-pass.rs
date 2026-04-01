//@ check-pass

fn main() {}

#[cfg(false)]
trait X {
    default const A: u8;
    default const B: u8 = 0;
    default type D;
    default type C: Ord;
    default fn f1();
    default fn f2() {}
}
