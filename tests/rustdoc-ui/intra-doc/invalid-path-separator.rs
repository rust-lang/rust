// This test shows that `broken_intra_doc_links` emits a more meaningful message when
// it encounters a path that is unresolvable due to being invalid.
#![deny(rustdoc::broken_intra_doc_links)]

//! [std:path]
//~^ ERROR
//
//! [std:::path]
//~^ ERROR
//
//! [std::::path]
//~^ ERROR
//
//! [std:::::path]
//~^ ERROR
//
//! [std2::::path]
//~^ ERROR
