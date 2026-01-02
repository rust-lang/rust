#![crate_name = "foo"]

pub trait Foo {
    type AssocType;

    const THE_CONST: Self::AssocType;
}

pub struct Bar;

impl Foo for Bar {
    type AssocType = (u32,);

    //@ has foo/struct.Bar.html
    //@ matches - '//section[@id="associatedconstant.THE_CONST"]/h4[@class="code-header"]' '^const THE_CONST: Self::AssocType'
    //@ !matches - '//section[@id="associatedconstant.THE_CONST"]//*[last()]' '{transmute'
    const THE_CONST: Self::AssocType = (1u32,);
}
