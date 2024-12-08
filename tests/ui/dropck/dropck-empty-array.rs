//@ run-pass

#[allow(dead_code)]
struct Struct<'s>(&'s str);

impl<'s> Drop for Struct<'s> {
    fn drop(&mut self) {}
}

fn to_array_zero<T>(_: T) -> [T; 0] {
    []
}

pub fn array_zero_in_tuple() {
    let mut x = ([], String::new());
    {
        let s = String::from("temporary");
        let p = Struct(&s);
        x.0 = to_array_zero(p);
    }
}

fn main() {}
