pub enum Attribute {
    Code {attr_name_idx: u16},
}

pub enum Foo {
    Bar
}

fn test(a: Foo) {
    println!("{}", a.b); //~ no field `b` on type `Foo`
}

fn main() {
    let x = Attribute::Code {
        attr_name_idx: 42,
    };
    let z = (&x).attr_name_idx; //~ no field `attr_name_idx` on type `&Attribute`
    let y = x.attr_name_idx; //~ no field `attr_name_idx` on type `Attribute`
}
