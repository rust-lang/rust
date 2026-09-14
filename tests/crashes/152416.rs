//@ known-bug: #152416
//@ needs-rustc-debug-assertions
//@ compile-flags: -Zunstable-options

trait AssetID {}
trait Archive<X> {
    fn name(&self);
}
struct NorthlightAssetID;
impl AssetID for NorthlightAssetID {}
fn get() -> Box<dyn Archive<impl AssetID>> {
    let x: Box<dyn Archive<NorthlightAssetID>> = todo!();
    x
}
fn main() {
    get().name();
}
