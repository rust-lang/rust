// https://github.com/rust-lang/rust/issues/24365
pub enum Attribute {
    Code {attr_name_idx: u16},
}

pub enum Foo {
    Bar
}

fn test(a: Foo) {
    println!("{}", a.b); //~ ERROR no field `b` on type `Foo`
}

fn main() {
    let x = Attribute::Code {
        attr_name_idx: 42,
    };
    let z = (&x).attr_name_idx; //~ ERROR no field `attr_name_idx` on type `&Attribute`
    let y = x.attr_name_idx; //~ ERROR no field `attr_name_idx` on type `Attribute`
}
