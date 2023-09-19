use autodiff::autodiff;

use std::io;

// Will be represented as {f32, i16, i16} when passed by reference
// will be represented as i64 if passed by value
struct Foo {
    c1: i16,
    a: f32,
    c2: i16,
}

#[autodiff(cos, Reverse, Active, Duplicated)]
fn sin(x: &Foo) -> f32 {
    assert!(x.c1 < x.c2);
    f32::sin(x.a)
}

fn main() {
    let mut s = String::new();
    println!("Please enter a value for c1");
    io::stdin().read_line(&mut s).unwrap();
    let c2 = s.trim_end().parse::<i16>().unwrap();
    dbg!(c2);

    let foo = Foo { c1: 4, a: 3.14, c2 };
    let mut df_dfoo = Foo { c1: 4, a: 0.0, c2 };

    dbg!(df_dfoo.a);
    dbg!(cos(&foo, &mut df_dfoo, 1.0));
    dbg!(df_dfoo.a);
    dbg!(f32::cos(foo.a));
}
