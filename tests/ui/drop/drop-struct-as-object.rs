//@ run-pass
#![allow(unused_variables)]
#![allow(non_upper_case_globals)]

// Test that destructor on a struct runs successfully after the struct
// is boxed and converted to an object.

// FIXME(static_mut_refs): this could use an atomic
#![allow(static_mut_refs)]

static mut value: usize = 0;

struct Cat {
    name : usize,
}

trait Dummy {
    fn get(&self) -> usize; //~ WARN method `get` is never used
}

impl Dummy for Cat {
    fn get(&self) -> usize { self.name }
}

impl Drop for Cat {
    fn drop(&mut self) {
        unsafe { value = self.name; }
    }
}

pub fn main() {
    {
        let x = Box::new(Cat {name: 22});
        let nyan: Box<dyn Dummy> = x as Box<dyn Dummy>;
    }
    unsafe {
        assert_eq!(value, 22);
    }
}
