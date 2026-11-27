use std::marker::PhantomData;

pub struct Foo<'a> {
    f: PhantomData<&'a u32>,
}

pub struct ContentType {
    pub ttype: Foo<'static>,
    pub subtype: Foo<'static>,
    pub params: Option<Foo<'static>>,
}

impl ContentType {
    //@ has const_doc/struct.ContentType.html
    //@ has  - '//*[@id="associatedconstant.Any"]' 'const Any: ContentType'
    pub const Any: ContentType = ContentType { ttype: Foo { f: PhantomData, },
                                               subtype: Foo { f: PhantomData, },
                                               params: None, };
}
