// run-pass
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
    println!("foo {}",fo);
}
