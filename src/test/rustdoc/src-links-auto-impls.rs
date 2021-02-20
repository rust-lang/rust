#![crate_name = "foo"]

// @has foo/struct.Unsized.html
// @has - '//h3[@id="impl-Sized"]/code' 'impl !Sized for Unsized'
// @!has - '//h3[@id="impl-Sized"]/a[@class="srclink"]' '[src]'
// @has - '//h3[@id="impl-Sync"]/code' 'impl Sync for Unsized'
// @!has - '//h3[@id="impl-Sync"]/a[@class="srclink"]' '[src]'
// @has - '//h3[@id="impl-Any"]/code' 'impl<T> Any for T'
// @has - '//h3[@id="impl-Any"]/a[@class="srclink"]' '[src]'
pub struct Unsized {
    data: [u8],
}
