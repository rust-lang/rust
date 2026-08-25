#![crate_type = "lib"]

#[link(name = "")]
//~^ ERROR link name must not be empty
unsafe extern "C" {
    #[link_name = ""]
    //~^ ERROR link name must not be empty
    safe fn empty();
}

#[link(name = "   ")]
unsafe extern "C" {
    #[link_name = "  "]
    safe fn this_is_fine();
}

#[export_name = "  "]
extern "C" fn bar() -> i32 {
    42
}

#[link(name = "\0")]
//~^ ERROR link name may not contain null characters
unsafe extern "C" {}

#[link(name = "foo\0")]
//~^ ERROR link name may not contain null characters
unsafe extern "C" {}

#[link(name = "\0foo")]
//~^ ERROR link name may not contain null characters
unsafe extern "C" {}

#[link(name = "fo\0o")]
//~^ ERROR link name may not contain null characters
unsafe extern "C" {}

unsafe extern "C" {
    #[link_name = "\0"]
    //~^ ERROR link name may not contain null characters
    safe fn empty_null();

    #[link_name = "foo\0"]
    //~^ ERROR link name may not contain null characters
    safe fn trailing_null();

    #[link_name = "\0foo"]
    //~^ ERROR link name may not contain null characters
    safe fn leading_null();

    #[link_name = "fo\0o"]
    //~^ ERROR link name may not contain null characters
    safe fn middle_null();
}
