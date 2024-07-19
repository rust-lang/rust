// 2015
type A = dyn Iterator<Item=Foo<'a>> + 'a;
type A = &dyn Iterator<Item=Foo<'a>> + 'a;
type A = dyn::Path;
