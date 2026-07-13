//@ check-pass

#[link_name = "VALUE"]
//~^ WARN `#[link_name]` attribute cannot be used on statics
//~| WARN this was previously accepted by the compiler but is being phased out
static VALUE_DEFINITION: u8 = 0;

unsafe extern "C" {
    #[link_name = "VALUE"]
    static VALUE_DECLARATION: u8;
}

fn main() {}
