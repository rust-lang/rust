//! Regression test for https://github.com/rust-lang/rust/issues/11740

//@ check-pass

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

fn main() {
    let element = Element { attrs: Vec::new() };
    unsafe { let () = element.get_attr("foo"); };
}
