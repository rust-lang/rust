#![crate_type = "lib"]

#![doc(extern_html_root_url)]
//~^ ERROR
#![doc(extern_html_root_url = "a")]
//~^ ERROR
#![doc(extern_html_root_url(a))]
//~^ ERROR
#![doc(extern_html_root_url(a()))]
//~^ ERROR
#![doc(extern_html_root_url(a = 1))]
//~^ ERROR
#![doc(extern_html_root_url(a = ""))]
//~^ ERROR
#![doc(extern_html_root_url(a = "1"))] // This one is supposed to work.
#![doc(extern_html_root_url(b = "2", c = "3"))] // This one is supposed to work.

pub fn foo() {}
