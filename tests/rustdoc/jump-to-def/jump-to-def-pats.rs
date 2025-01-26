// This test ensures that patterns also get a link generated.

//@ compile-flags: -Zunstable-options --generate-link-to-definition

#![crate_name = "foo"]

//@ has 'src/foo/jump-to-def-pats.rs.html'

use std::fmt;

pub enum MyEnum<T, E> {
    Ok(T),
    Err(E),
    Some(T),
    None,
}

pub enum X {
    A,
}

pub fn foo() -> Result<(), ()> {
    // FIXME: would be nice to be able to check both the class and the href at the same time so
    // we could check the text as well...
    //@ has - '//a[@class="prelude-val"]/@href' '{{channel}}/core/result/enum.Result.html#variant.Ok'
    //@ has - '//a[@href="{{channel}}/core/result/enum.Result.html#variant.Ok"]' 'Ok'
    Ok(())
}

impl<T, E> fmt::Display for MyEnum<T, E> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            //@ has - '//a[@href="#12"]' 'Ok'
            Self::Ok(_) => f.write_str("MyEnum::Ok"),
            //@ has - '//a[@href="#13"]' 'Err'
            MyEnum::Err(_) => f.write_str("MyEnum::Err"),
            //@ has - '//a[@href="#14"]' 'Some'
            Self::Some(_) => f.write_str("MyEnum::Some"),
            //@ has - '//a[@href="#15"]' 'None'
            Self::None => f.write_str("MyEnum::None"),
        }
    }
}

impl X {
    fn p(&self) -> &str {
        match self {
            //@ has - '//a[@href="#19"]' 'A'
            Self::A => "X::A",
        }
    }
}
