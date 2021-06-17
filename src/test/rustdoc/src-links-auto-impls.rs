#![crate_name = "foo"]

// @has foo/struct.Unsized.html
// @has - '//div[@id="impl-Sized"]/code' 'impl !Sized for Unsized'
// @!has - '//div[@id="impl-Sized"]//a[@class="srclink"]' '[src]'
// @has - '//div[@id="impl-Sync"]/code' 'impl Sync for Unsized'
// @!has - '//div[@id="impl-Sync"]//a[@class="srclink"]' '[src]'
// @has - '//div[@id="impl-Any"]/code' 'impl<T> Any for T'
// @has - '//div[@id="impl-Any"]//a[@class="srclink"]' '[src]'
pub struct Unsized {
    data: [u8],
}
