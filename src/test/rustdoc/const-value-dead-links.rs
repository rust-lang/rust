// aux-crate:aux=const-value.rs
// edition:2021
#![crate_name = "consts"]

// Test that *no* hyperlink anchors are created for the structs here since their
// documentation wasn't built (the dependency crate was only *compiled*).
// Re snapshots: Check that this is indeed the case.
//
// NB: The corresponding test cases where the docs of the dependency *were* built
//     can be found in `./const-value.rs` (as `Struct::{DATA, OPAQ}`).

// @has 'consts/constant.DATA.html'
// @has - '//*[@class="docblock item-decl"]//code' \
//        'const DATA: Data = Data { open: (0, 0, 1), .. }'
// @snapshot data - '//*[@class="docblock item-decl"]//code'
pub const DATA: aux::Data = aux::Data::new((0, 0, 1));

// @has 'consts/constant.OPAQ.html'
// @has - '//*[@class="docblock item-decl"]//code' \
//        'const OPAQ: Opaque = Opaque(_)'
// @snapshot opaq - '//*[@class="docblock item-decl"]//code'
pub const OPAQ: aux::Opaque = aux::Opaque::new(0xff00);
