#![crate_name = "foo"]

// @has foo/struct.Unsized.html
// @has - '//div[@id="impl-Sized"]/h3[@class="code-header in-band"]' 'impl !Sized for Unsized'
// @!has - '//div[@id="impl-Sized"]//a[@class="srclink"]' '[src]'
// @has - '//div[@id="impl-Sync"]/h3[@class="code-header in-band"]' 'impl Sync for Unsized'
// @!has - '//div[@id="impl-Sync"]//a[@class="srclink"]' '[src]'
// @has - '//div[@id="impl-Any"]/h3[@class="code-header in-band"]' 'impl<T> Any for T'
// @has - '//div[@id="impl-Any"]//a[@class="srclink"]' '[src]'
pub struct Unsized {
    data: [u8],
}
