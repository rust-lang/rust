//@ run-pass
// allows aligned custom discriminant enums to cast into other types
// See the issue #92464 for more info
#[allow(dead_code)]
#[repr(align(8))]
enum Aligned {
    Zero = 0,
    One = 1,
}

fn main() {
    let aligned = Aligned::Zero;
    let fo = aligned as u8;
    println!("foo {}", fo);
    assert_eq!(fo, 0);
    println!("{}", tou8(Aligned::Zero));
    assert_eq!(tou8(Aligned::Zero), 0);
}

#[inline(never)]
fn tou8(al: Aligned) -> u8 {
    // Cast behind a function call so ConstProp does not see it
    // (so that we can test codegen).
    al as u8
}
