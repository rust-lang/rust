//@ edition: 2021
//@ aux-build: macro-generated-extern-crate.rs

const _: () = {
    extern crate macro_generated_extern_crate as _my_crate;
    impl _my_crate::MyTrait for Local {}
};

struct Local;

fn main() {
    Local::custom();
    //~^ ERROR no associated function or constant named `custom` found for struct `Local`
}
