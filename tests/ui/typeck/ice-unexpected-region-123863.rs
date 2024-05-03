const fn concat_strs<const A: &'static str>() -> &'static str {
//~^ ERROR &'static str` is forbidden as the type of a const generic parameter
    struct Inner<const A: &'static str>;
//~^ ERROR &'static str` is forbidden as the type of a const generic parameter
    Inner::concat_strs::<"a">::A
//~^ ERROR ambiguous associated type
}

pub fn main() {}
