#[link_name = "VALUE"]
//~^ WARN the `link_name` attribute cannot be used on statics
//~| WARN this was previously accepted by the compiler but is being phased out
static VALUE_DEFINITION: u8 = 0;

#[unsafe(link_name = "UNSAFE_VALUE")]
//~^ ERROR `link_name` is not an unsafe attribute
//~| WARN the `link_name` attribute cannot be used on statics
//~| WARN this was previously accepted by the compiler but is being phased out
static UNSAFE_VALUE_DEFINITION: u8 = 0;

unsafe extern "C" {
    #[link_name = "VALUE"]
    static VALUE_DECLARATION: u8;
}

fn main() {}
