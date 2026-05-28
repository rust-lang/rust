//@build-pass
// Some code that looks like it might be relying on promotion, but actually this is using the
// enclosing-scope rule, meaning the reference is "extended" to outlive its block and live as long
// as the surrounding block (which in this case is the entire program). There are multiple
// allocations being interned at once.

struct Gen<T>(T);
impl<'a, T> Gen<&'a T> {
    // Can't be promoted because `T` might not be `'static`.
    const C: &'a [T] = &[];
}

// Can't be promoted because of `Drop`.
const V: &Vec<i32> = &Vec::new();

fn main() {}
