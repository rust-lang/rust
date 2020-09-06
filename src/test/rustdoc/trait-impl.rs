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
    // @has trait_impl/struct.Struct.html '//*[@id="method.a"]/../div/p' 'Some long docs'
    // @!has - '//*[@id="method.a"]/../div/p' 'link will be added'
    // @has - '//*[@id="method.a"]/../div/p/a' 'Read more'
    // @has - '//*[@id="method.a"]/../div/p/a/@href' 'trait.Trait.html'
    fn a() {}

    // @has trait_impl/struct.Struct.html '//*[@id="method.b"]/../div/p' 'These docs contain'
    // @has - '//*[@id="method.b"]/../div/p/a' 'reference link'
    // @has - '//*[@id="method.b"]/../div/p/a/@href' 'https://example.com'
    // @has - '//*[@id="method.b"]/../div/p/a' 'Read more'
    // @has - '//*[@id="method.b"]/../div/p/a/@href' 'trait.Trait.html'
    fn b() {}

    // @!has trait_impl/struct.Struct.html '//*[@id="method.c"]/../div/p' 'code block'
    // @has - '//*[@id="method.c"]/../div/p/a' 'Read more'
    // @has - '//*[@id="method.c"]/../div/p/a/@href' 'trait.Trait.html'
    fn c() {}

    // @has trait_impl/struct.Struct.html '//*[@id="method.d"]/../div/p' \
    //   'Escaped formatting a*b*c* works'
    // @!has trait_impl/struct.Struct.html '//*[@id="method.d"]/../div/p/em'
    fn d() {}
}
