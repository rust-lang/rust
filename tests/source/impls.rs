impl Foo for Bar { fn foo() { "hi" } }

pub impl Foo for Bar {
    // Associated Constants
    const   Baz:   i32 =   16;
    // Associated Types
    type   FooBar  =   usize;
    // Comment 1
    fn foo() { "hi" }
    // Comment 2
    fn foo() { "hi" }
    // Comment 3
}

pub unsafe impl<'a, 'b, X, Y: Foo<Bar>> !Foo<'a, X> for Bar<'b, Y> where X: Foo<'a, Z> {
    fn foo() { "hi" }    
}

impl<'a, 'b, X, Y: Foo<Bar>> Foo<'a, X> for Bar<'b, Y> where X: Fooooooooooooooooooooooooooooo<'a, Z>
{
    fn foo() { "hi" }    
}

impl<'a, 'b, X, Y: Foo<Bar>> Foo<'a, X> for Bar<'b, Y> where X: Foooooooooooooooooooooooooooo<'a, Z>
{
    fn foo() { "hi" }    
}

impl<T> Foo for Bar<T> where T: Baz 
{
}

impl<T> Foo for Bar<T> where T: Baz { /* Comment */ }

impl Foo {
    fn foo() {}
}

impl Boo {

    // BOO
    fn boo() {}
    // FOO

    
    
}

mod a {
    impl Foo {
        // Hello!
        fn foo() {}
    }
}


mod b {
    mod a {
        impl Foo {
            fn foo() {}
        }
    }
}

impl Foo { add_fun!(); }

impl Blah {
    fn boop() {}
    add_fun!();
}

impl X { fn do_parse(  mut  self : X ) {} }

impl Y5000 {
    fn bar(self: X< 'a ,  'b >, y: Y) {}

    fn bad(&self, ( x, y): CoorT) {}

    fn turbo_bad(self: X< 'a ,  'b >  , ( x, y): CoorT) {
        
    }
}

pub impl<T> Foo for Bar<T> where T: Foo
{
    fn foo() { "hi" }
}

pub impl<T, Z> Foo for Bar<T, Z> where T: Foo, Z: Baz {}

mod m {
    impl<T> PartialEq for S<T> where T: PartialEq {
        fn eq(&self, other: &Self) {
            true
        }
      }

        impl<T> PartialEq for S<T> where T: PartialEq {      }
 }

impl<BorrowType, K, V, NodeType, HandleType> Handle<NodeRef<BorrowType, K, V, NodeType>, HandleType> {
}
