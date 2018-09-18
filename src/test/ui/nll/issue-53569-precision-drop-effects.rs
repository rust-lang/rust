#![feature(nll)]
#![allow(dead_code)]

// compile-pass

// rust-lang/rust#53569: a drop imposed solely by one enum variant
// should not conflict with a reborrow from another variant of that
// enum.

struct S<'a> {
    value: &'a Value,
}

struct Value {
    data: u32,
}

impl<'a> S<'a> {
    fn get(&self) -> Result<&'a mut Value, String> {
        unimplemented!();
    }

    fn at(&self)  {
        let v = self.get();
        if let Ok(Value { ref mut data }) = v {
            let _res : &'a u32 = data;
        }
    }
}

fn main() {
}
