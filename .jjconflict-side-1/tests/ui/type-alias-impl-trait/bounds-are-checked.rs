// Make sure that we check that impl trait types implement the traits that they
// claim to.

#![feature(type_alias_impl_trait)]

type X<'a> = impl Into<&'static str> + From<&'a str>;

#[define_opaque(X)]
fn f<'a: 'static>(t: &'a str) -> X<'a> {
    t
    //~^ ERROR expected generic lifetime parameter, found `'static`
}

fn extend_lt<'a>(o: &'a str) -> &'static str {
    X::<'_>::from(o).into()
}

fn main() {
    let r = {
        let s = "abcdef".to_string();
        extend_lt(&s)
    };
    println!("{}", r);
}
