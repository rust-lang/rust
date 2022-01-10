#![crate_name = "foo"]

// @has foo/struct.Unsized.html
// @has - '//div[@id="impl-Sized-for-Unsized"]/h3[@class="code-header in-band"]' 'impl !Sized for Unsized'
// @!has - '//*[@id="impl-Sized"]//a[@class="srclink"]' 'source'
// @has - '//*[@id="impl-Sync-for-Unsized"]/h3[@class="code-header in-band"]' 'impl Sync for Unsized'
// @!has - '//*[@id="impl-Sync"]//a[@class="srclink"]' 'source'
// @has - '//*[@id="impl-Any-for-Unsized"]/h3[@class="code-header in-band"]' 'impl<T> Any for T'
// @has - '//*[@id="impl-Any-for-Unsized"]//a[@class="srclink"]' 'source'
pub struct Unsized {
    data: [u8],
}
