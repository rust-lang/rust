#[no_mangle]
fn foo() {}
//~^ HELP: it's first defined here, in crate `exported_symbol_clashing`

#[export_name = "foo"]
fn bar() {}
//~^ HELP: then it's defined here again, in crate `exported_symbol_clashing`

fn main() {
    extern "Rust" {
        fn foo();
    }
    unsafe { foo() }
    //~^ ERROR: multiple definitions of symbol `foo`
}
