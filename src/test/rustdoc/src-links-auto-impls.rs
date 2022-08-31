#![crate_name = "foo"]

// @has foo/struct.Unsized.html
// @has - '//*[@id="impl-Sized-for-Unsized"]/h3[@class="code-header in-band"]' 'impl !Sized for Unsized'
// @!has - '//*[@id="impl-Sized-for-Unsized"]//a[@class="srclink"]' 'source'
// @has - '//*[@id="impl-Sync-for-Unsized"]/h3[@class="code-header in-band"]' 'impl Sync for Unsized'
// @!has - '//*[@id="impl-Sync-for-Unsized"]//a[@class="srclink"]' 'source'
// @has - '//*[@id="impl-Any-for-Unsized"]/h3[@class="code-header in-band"]' 'impl<T> Any for T'
// @has - '//*[@id="impl-Any-for-Unsized"]//a[@class="srclink rightside"]' 'source'
pub struct Unsized {
    data: [u8],
}
