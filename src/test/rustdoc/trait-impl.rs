pub trait Trait {
    /// Some long docs here.
    ///
    /// These docs are long enough that a link will be added to the end.
    fn a();

    /// These docs contain a [reference link].
    ///
    /// [reference link]: https://example.com
    fn b();

    /// ```
    /// This code block should not be in the output, but a Read more link should be generated
    /// ```
    fn c();

    /// Escaped formatting a\*b\*c\* works
    fn d();
}

pub struct Struct;

impl Trait for Struct {
    // @has trait_impl/struct.Struct.html '//*[@id="method.a"]/../../div[@class="docblock"]/p' 'Some long docs'
    // @!has - '//*[@id="method.a"]/../../div[@class="docblock"]/p' 'link will be added'
    // @has - '//*[@id="method.a"]/../../div[@class="docblock"]/p/a' 'Read more'
    // @has - '//*[@id="method.a"]/../../div[@class="docblock"]/p/a/@href' 'trait.Trait.html#tymethod.a'
    fn a() {}

    // @has - '//*[@id="method.b"]/../../div[@class="docblock"]/p' 'These docs contain'
    // @has - '//*[@id="method.b"]/../../div[@class="docblock"]/p/a' 'reference link'
    // @has - '//*[@id="method.b"]/../../div[@class="docblock"]/p/a/@href' 'https://example.com'
    // @has - '//*[@id="method.b"]/../../div[@class="docblock"]/p/a' 'Read more'
    // @has - '//*[@id="method.b"]/../../div[@class="docblock"]/p/a/@href' 'trait.Trait.html#tymethod.b'
    fn b() {}

    // @!has - '//*[@id="method.c"]/../../div[@class="docblock"]/p' 'code block'
    // @has - '//*[@id="method.c"]/../../div[@class="docblock"]/a' 'Read more'
    // @has - '//*[@id="method.c"]/../../div[@class="docblock"]/a/@href' 'trait.Trait.html#tymethod.c'
    fn c() {}

    // @has - '//*[@id="method.d"]/../../div[@class="docblock"]/p' 'Escaped formatting a*b*c* works'
    // @!has - '//*[@id="method.d"]/../../div[@class="docblock"]/p/em'
    fn d() {}

    // @has - '//*[@id="impl-Trait"]/code/a/@href' 'trait.Trait.html'
}
