// This test ensures that negative impls don't have items listed inside them.

#![feature(negative_impls)]
#![crate_name = "foo"]

pub struct Thing;

//@ has 'foo/struct.Thing.html'
// We check the full path to ensure there is no `<details>` element.
//@ has - '//div[@id="trait-implementations-list"]/section[@id="impl-Iterator-for-Thing"]/h3' \
// 'impl !Iterator for Thing'
impl !Iterator for Thing {}

// This struct will allow us to compare both paths.
pub struct Witness;

//@ has 'foo/struct.Witness.html'
//@ has - '//div[@id="trait-implementations-list"]/details//section[@id="impl-Iterator-for-Witness"]/h3' \
// 'impl Iterator for Witness'
impl Iterator for Witness {
    type Item = u8;

    fn next(&mut self) -> Option<Self::Item> {
        None
    }
}
