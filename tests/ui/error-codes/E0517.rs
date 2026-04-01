#[repr(C)] //~ ERROR: E0517
type Foo = u8;

#[repr(packed)] //~ ERROR: E0517
enum Foo2 {Bar, Baz}

#[repr(u8)] //~ ERROR: E0517
struct Foo3 {bar: bool, baz: bool}

#[repr(C)] //~ ERROR: E0517
impl Foo3 {
}

fn main() {
}
