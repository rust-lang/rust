struct Binder(i32, i32, i32);

fn main() {
    let x = Binder(1, 2, 3);
    match x {
        Binder(_a, _x @ ..) => {}
        _ => {}
    }
}
//~^^^^ ERROR `_x @` is not allowed in a tuple struct
//~| ERROR: `..` patterns are not allowed here
//~| ERROR: this pattern has 2 fields, but the corresponding tuple struct has 3 fields
