extern crate proc_macro;
use proc_macro::*;

#[proc_macro]
pub fn gen_macro_rules(_: TokenStream) -> TokenStream {
    "
    macro_rules! generated {() => {
        struct ItemDef;
        let local_def = 0;

        ItemUse; // OK
        local_use; // ERROR
        break 'label_use; // ERROR

        type DollarCrate = $crate::ItemUse; // OK
    }}
    ".parse().unwrap()
}
