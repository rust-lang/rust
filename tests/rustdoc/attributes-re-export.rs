// Tests that attributes are correctly copied onto a re-exported item.
//@ edition:2021
#![crate_name = "re_export"]

//@ has 're_export/fn.thingy2.html' '//pre[@class="rust item-decl"]' '#[unsafe(no_mangle)]'
pub use thingymod::thingy as thingy2;

mod thingymod {
    #[no_mangle]
    pub fn thingy() {

    }
}
