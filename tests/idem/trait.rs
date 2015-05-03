// Test traits

trait Foo {
    fn bar(x: i32) -> Baz<U> {
        Baz::new()
    }

    fn baz(a: AAAAAAAAAAAAAAAAAAAAAA, b: BBBBBBBBBBBBBBBBBBBBBBBBBBBBBBB) -> RetType;

    fn foo(a: AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA, // Another comment
           b: BBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBB)
           -> RetType; // Some comment

    fn baz(&mut self) -> i32;

    fn increment(&mut self, x: i32);

    fn read(&mut self, x: BufReader<R> /* Used to be MemReader */)
        where R: Read;
}
