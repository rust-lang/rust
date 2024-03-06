//@ aux-build:issue-73112.rs

extern crate issue_73112;

fn main() {
    use issue_73112::PageTable;

    #[repr(C, packed)]
    struct SomeStruct {
    //~^ ERROR packed type cannot transitively contain a `#[repr(align)]` type [E0588]
        page_table: PageTable,
    }
}
