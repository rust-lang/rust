// compile-flags: -Z parse-only

struct Foo<'a, 'b> {
    a: &'a &'b i32
}

fn foo<'a, 'b>(x: &mut Foo<'a; 'b>) {}
//~^ ERROR expected one of `,` or `>`, found `;`
