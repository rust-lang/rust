#![crate_name = "foo"]

//@ has foo/struct.Unsized.html
//@ has - '//*[@id="impl-Sized-for-Unsized"]/h3[@class="code-header"]' 'impl !Sized for Unsized'
//@ !has - '//*[@id="impl-Sized-for-Unsized"]//a[@class="src"]' 'Source'
//@ has - '//*[@id="impl-Sync-for-Unsized"]/h3[@class="code-header"]' 'impl Sync for Unsized'
//@ !has - '//*[@id="impl-Sync-for-Unsized"]//a[@class="src"]' 'Source'
//@ has - '//*[@id="impl-Any-for-T"]/h3[@class="code-header"]' 'impl<T> Any for T'
//@ has - '//*[@id="impl-Any-for-T"]//a[@class="src rightside"]' 'Source'
pub struct Unsized {
    data: [u8],
}
