// https://github.com/rust-lang/rust/issues/73112
//@ aux-build:aux-73112.rs

extern crate aux_73112;

fn main() {
    use aux_73112::PageTable;

    #[repr(C, packed)]
    struct SomeStruct {
    //~^ ERROR packed type cannot transitively contain a `#[repr(align)]` type [E0588]
        page_table: PageTable,
    }
}
