#![feature(rustc_attrs)]
#![allow(warnings)]

struct Attr {
    name: String,
    value: String,
}

struct Element {
    attrs: Vec<Box<Attr>>,
}

impl Element {
    pub unsafe fn get_attr<'a>(&'a self, name: &str) {
        self.attrs
            .iter()
            .find(|attr| {
                      let attr: &&Box<Attr> = std::mem::transmute(attr);
                      true
                  });
    }
}

#[rustc_error]
fn main() { //~ ERROR compilation successful
    let element = Element { attrs: Vec::new() };
    let _ = unsafe { element.get_attr("foo") };
}
