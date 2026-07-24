#[unsafe(link_name = "VALUE")]
//~^ ERROR `link_name` is not an unsafe attribute
//~| WARN `#[link_name]` attribute cannot be used on statics
//~| WARN this was previously accepted by the compiler but is being phased out
static VALUE_DEFINITION: u8 = 0;

fn main() {}
