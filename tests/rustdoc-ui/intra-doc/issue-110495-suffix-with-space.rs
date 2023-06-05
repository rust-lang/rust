// this test used to ICE
#![deny(rustdoc::broken_intra_doc_links)]
//! [Clone ()]. //~ ERROR unresolved
//! [Clone !]. //~ ERROR incompatible
//! [`Clone ()`]. //~ ERROR unresolved
//! [`Clone !`]. //~ ERROR incompatible
