// check-pass
// revisions: mir thir
// [thir]compile-flags: -Zthir-unsafeck

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
