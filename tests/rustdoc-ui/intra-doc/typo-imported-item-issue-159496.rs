// Suggest similarly named module items, including imports, for unresolved intra-doc links.

#![deny(rustdoc::broken_intra_doc_links)]

use std::collections::HashMap;

/// Creates a [Hashmap].
//~^ ERROR unresolved link to `Hashmap`
pub fn create() {
    let _: Option<HashMap<(), ()>> = None;
}

/// Creates a [std::collections::Hashmap].
//~^ ERROR unresolved link to `std::collections::Hashmap`
pub fn qualified() {}

/// Creates a [type@Hashmap].
//~^ ERROR unresolved link to `Hashmap`
pub fn disambiguated() {}

/// Creates a [`Hashmap<Hashmap>`].
//~^ ERROR unresolved link to `Hashmap`
pub fn generic() {}
