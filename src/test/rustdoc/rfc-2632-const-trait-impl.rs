// Test that we do not currently display `~const` in rustdoc
// as that syntax is currently provisional; `~const Drop` has
// no effect on stable code so it should be hidden as well.
//
// To future blessers: make sure that `const_trait_impl` is
// stabilized when changing `@!has` to `@has`, and please do
// not remove this test.
#![feature(const_trait_impl)]
#![crate_name = "foo"]

pub struct S<T>(T);

// @!has foo/trait.Tr.html '//pre[@class="rust trait"]/code/a[@class="trait"]' '~const'
// @!has - '//pre[@class="rust trait"]/code/a[@class="trait"]' 'Drop'
// @has - '//pre[@class="rust trait"]/code/a[@class="trait"]' 'Clone'
// @!has - '//pre[@class="rust trait"]/code/span[@class="where"]' '~const'
// @!has - '//pre[@class="rust trait"]/code/span[@class="where"]' 'Drop'
// @has - '//pre[@class="rust trait"]/code/span[@class="where"]' ': Clone'
pub trait Tr<T> {
    // @!has - '//div[@id="method.a"]/h4[@class="code-header"]' '~const'
    // @!has - '//div[@id="method.a"]/h4[@class="code-header"]/a[@class="trait"]' 'Drop'
    // @has - '//div[@id="method.a"]/h4[@class="code-header"]/a[@class="trait"]' 'Clone'
    // @!has - '//div[@id="method.a"]/h4[@class="code-header"]/span[@class="where"]' '~const'
    // @!has - '//div[@id="method.a"]/h4[@class="code-header"]/span[@class="where fmt-newline"]' 'Drop'
    // @has - '//div[@id="method.a"]/h4[@class="code-header"]/span[@class="where fmt-newline"]' ': Clone'
    #[default_method_body_is_const]
    fn a<A: ~const Drop + ~const Clone>() where Option<A>: ~const Drop + ~const Clone {}
}

// @!has - '//section[@id="impl-Tr%3CT%3E"]/h3[@class="code-header in-band"]' '~const'
// @!has - '//section[@id="impl-Tr%3CT%3E"]/h3[@class="code-header in-band"]/a[@class="trait"]' 'Drop'
// @has - '//section[@id="impl-Tr%3CT%3E"]/h3[@class="code-header in-band"]/a[@class="trait"]' 'Clone'
// @!has - '//section[@id="impl-Tr%3CT%3E"]/h3[@class="code-header in-band"]/span[@class="where"]' '~const'
// @!has - '//section[@id="impl-Tr%3CT%3E"]/h3[@class="code-header in-band"]/span[@class="where fmt-newline"]' 'Drop'
// @has - '//section[@id="impl-Tr%3CT%3E"]/h3[@class="code-header in-band"]/span[@class="where fmt-newline"]' ': Clone'
impl<T: ~const Drop + ~const Clone> const Tr<T> for T where Option<T>: ~const Drop + ~const Clone {
    fn a<A: ~const Drop + ~const Clone>() where Option<A>: ~const Drop + ~const Clone {}
}

// @!has foo/fn.foo.html '//pre[@class="rust fn"]/code/a[@class="trait"]' '~const'
// @!has - '//pre[@class="rust fn"]/code/a[@class="trait"]' 'Drop'
// @has - '//pre[@class="rust fn"]/code/a[@class="trait"]' 'Clone'
// @!has - '//pre[@class="rust fn"]/code/span[@class="where fmt-newline"]' '~const'
// @!has - '//pre[@class="rust fn"]/code/span[@class="where fmt-newline"]' 'Drop'
// @has - '//pre[@class="rust fn"]/code/span[@class="where fmt-newline"]' ': Clone'
pub const fn foo<F: ~const Drop + ~const Clone>() where Option<F>: ~const Drop + ~const Clone {
    F::a()
}

impl<T> S<T> {
    // @!has foo/struct.S.html '//section[@id="method.foo"]/h4[@class="code-header"]' '~const'
    // @!has - '//section[@id="method.foo"]/h4[@class="code-header"]/a[@class="trait"]' 'Drop'
    // @has - '//section[@id="method.foo"]/h4[@class="code-header"]/a[@class="trait"]' 'Clone'
    // @!has - '//section[@id="method.foo"]/h4[@class="code-header"]/span[@class="where"]' '~const'
    // @!has - '//section[@id="method.foo"]/h4[@class="code-header"]/span[@class="where fmt-newline"]' 'Drop'
    // @has - '//section[@id="method.foo"]/h4[@class="code-header"]/span[@class="where fmt-newline"]' ': Clone'
    pub const fn foo<B: ~const Drop + ~const Clone>() where B: ~const Drop + ~const Clone {
        B::a()
    }
}
