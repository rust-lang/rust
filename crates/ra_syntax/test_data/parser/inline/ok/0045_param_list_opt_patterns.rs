fn foo<F: FnMut(&mut Foo<'a>)>(){}
fn foo<F: FnMut(#[attr] &mut Foo<'a>)>(){}
