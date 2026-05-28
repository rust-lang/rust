// This test was triggering a `span_bug` crash, which was then fixed by
// downgrading it to a `span_delayed_bug`.

pub struct MyStruct<'field> {
    field: &'field [u32],
}

impl MyStruct<'_> {
    pub fn f(field: &[u32]) -> Self<u32> { //~ ERROR type arguments are not allowed on self type
        Self { field }                     //~ ERROR lifetime may not live long enough
    }
}

fn main() {}
