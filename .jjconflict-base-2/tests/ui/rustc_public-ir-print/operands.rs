//@ compile-flags: -Z unpretty=stable-mir --crate-type lib -C panic=abort -Zmir-opt-level=0
//@ check-pass
//@ only-x86_64
//@ needs-unwind unwind edges are different with panic=abort
//! Check how stable mir pretty printer prints different operands and abort strategy.

pub fn operands(val: u8) {
    let array = [val; 10];
    let first = array[0];
    let last = array[10 - 1];
    assert_eq!(first, last);

    let reference = &first;
    let dereferenced = *reference;
    assert_eq!(dereferenced, first);

    let tuple = (first, last);
    let (first_again, _) = tuple;
    let first_again_again = tuple.0;
    assert_eq!(first_again, first_again_again);

    let length = array.len();
    let size_of = std::mem::size_of_val(&length);
    assert_eq!(length, size_of);
}

pub struct Dummy {
    c: char,
    i: i32,
}

pub enum Ctors {
    Unit,
    StructLike { d: Dummy },
    TupLike(bool),
}

pub fn more_operands() -> [Ctors; 3] {
    let dummy = Dummy { c: 'a', i: i32::MIN };
    let unit = Ctors::Unit;
    let struct_like = Ctors::StructLike { d: dummy };
    let tup_like = Ctors::TupLike(false);
    [unit, struct_like, tup_like]
}

pub fn closures(x: bool, z: bool) -> impl FnOnce(bool) -> bool {
    move |y: bool| (x ^ y) || z
}
