// Test to ensure that `&` are handled the same way for generics and
// for other "normal" types.

#![crate_name = "foo"]

// @has 'foo/trait.Trait.html'

pub struct Struct;

pub trait Trait<Rhs = Self> {
    // @count - '//*[@id="tymethod.method"]/*[@class="code-header"]/a' 2
    // @has - '//*[@id="tymethod.method"]/*[@class="code-header"]/a' 'method'
    // @has - '//*[@id="tymethod.method"]/*[@class="code-header"]/a' '&'
    fn method(&self, other: &Rhs);
    // @count - '//*[@id="tymethod.method2"]/*[@class="code-header"]/a' 2
    // @has - '//*[@id="tymethod.method2"]/*[@class="code-header"]/a' 'method2'
    // @has - '//*[@id="tymethod.method2"]/*[@class="code-header"]/a' '*const'
    fn method2(&self, other: *const Rhs);
    // There should be only one `<a>` (just the method).
    // @count - '//*[@id="tymethod.bar"]/*[@class="code-header"]/a' 1
    // @has - '//*[@id="tymethod.bar"]/*[@class="code-header"]/a' 'bar'
    fn bar(&self, other: Rhs);
    // There should be two `<a>` (method and `Struct`).
    // @count - '//*[@id="tymethod.foo"]/*[@class="code-header"]/a' 2
    // @has - '//*[@id="tymethod.foo"]/*[@class="code-header"]/a' 'foo'
    // @has - '//*[@id="tymethod.foo"]/*[@class="code-header"]/a' 'Struct'
    fn foo(&self, other: &Struct);
    // There should be three `<a>` (method, `Struct` and `*const`).
    // @count - '//*[@id="tymethod.foo2"]/*[@class="code-header"]/a' 3
    // @has - '//*[@id="tymethod.foo2"]/*[@class="code-header"]/a' 'foo2'
    // @has - '//*[@id="tymethod.foo2"]/*[@class="code-header"]/a' 'Struct'
    // @has - '//*[@id="tymethod.foo2"]/*[@class="code-header"]/a' '*const'
    fn foo2(&self, other: *const Struct);
    // There should be only one `<a>` (just the method).
    // @count - '//*[@id="tymethod.tuple"]/*[@class="code-header"]/a' 1
    // @has - '//*[@id="tymethod.tuple"]/*[@class="code-header"]/a' 'tuple'
    fn tuple(&self, other: (Rhs, Rhs));
    // There should be two `<a>` (method and `Struct`).
    // @count - '//*[@id="tymethod.tuple2"]/*[@class="code-header"]/a' 2
    // @has - '//*[@id="tymethod.tuple2"]/*[@class="code-header"]/a' 'tuple2'
    // @has - '//*[@id="tymethod.tuple2"]/*[@class="code-header"]/a' 'Struct'
    fn tuple2(&self, other: (Struct, Rhs));
    // @count - '//*[@id="tymethod.slice"]/*[@class="code-header"]/a' 2
    // @has - '//*[@id="tymethod.slice"]/*[@class="code-header"]/a' 'slice'
    // @has - '//*[@id="tymethod.slice"]/*[@class="code-header"]/a' '&[Rhs]'
    fn slice(&self, other: &[Rhs]);
    // @count - '//*[@id="tymethod.ref_array"]/*[@class="code-header"]/a' 2
    // @has - '//*[@id="tymethod.ref_array"]/*[@class="code-header"]/a' 'ref_array'
    // @has - '//*[@id="tymethod.ref_array"]/*[@class="code-header"]/a' '[Rhs; 2]'
    fn ref_array(&self, other: &[Rhs; 2]);
    // There should be two `<a>` (method and `Struct`).
    // @count - '//*[@id="tymethod.slice2"]/*[@class="code-header"]/a' 2
    // @has - '//*[@id="tymethod.slice2"]/*[@class="code-header"]/a' 'slice2'
    // @has - '//*[@id="tymethod.slice2"]/*[@class="code-header"]/a' 'Struct'
    fn slice2(&self, other: &[Struct]);
    // @count - '//*[@id="tymethod.array"]/*[@class="code-header"]/a' 2
    // @has - '//*[@id="tymethod.array"]/*[@class="code-header"]/a' 'array'
    // @has - '//*[@id="tymethod.array"]/*[@class="code-header"]/a' '[Rhs; 2]'
    fn array(&self, other: [Rhs; 2]);
    // @count - '//*[@id="tymethod.array2"]/*[@class="code-header"]/a' 2
    // @has - '//*[@id="tymethod.array2"]/*[@class="code-header"]/a' 'array2'
    // @has - '//*[@id="tymethod.array2"]/*[@class="code-header"]/a' 'Struct'
    fn array2(&self, other: [Struct; 2]);
}
