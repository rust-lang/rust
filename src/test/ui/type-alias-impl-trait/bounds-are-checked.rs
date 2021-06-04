// Make sure that we check that impl trait types implement the traits that they
// claim to.

// revisions: min_tait full_tait
#![feature(min_type_alias_impl_trait)]
#![cfg_attr(full_tait, feature(type_alias_impl_trait))]
//[full_tait]~^ WARN incomplete

type X<'a> = impl Into<&'static str> + From<&'a str>;
//~^ ERROR mismatched types

fn f<'a: 'static>(t: &'a str) -> X<'a> {
    //~^ WARNING unnecessary lifetime parameter
    t
}

fn extend_lt<'a>(o: &'a str) -> &'static str {
    X::<'_>::from(o).into()
}

fn main() {
    let r =
    {
        let s = "abcdef".to_string();
        extend_lt(&s)
    };
    println!("{}", r);
}
