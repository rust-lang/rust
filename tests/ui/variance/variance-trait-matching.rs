#![allow(dead_code)]

// Get<T> is covariant in T
trait Get<T> {
    fn get(&self) -> T;
}

struct Cloner<T:Clone> {
    t: T
}

impl<T:Clone> Get<T> for Cloner<T> {
    fn get(&self) -> T {
        self.t.clone()
    }
}

fn get<'a, G>(get: &G) -> i32
    where G : Get<&'a i32>
{
    // This fails to type-check because, without variance, we can't
    // use `G : Get<&'a i32>` as evidence that `G : Get<&'b i32>`,
    // even if `'a : 'b`.
    pick(get, &22) //~ ERROR explicit lifetime required in the type of `get` [E0621]
}

fn pick<'b, G>(get: &'b G, if_odd: &'b i32) -> i32
    where G : Get<&'b i32>
{
    let v = *get.get();
    if v % 2 != 0 { v } else { *if_odd }
}

fn main() {
    let x = Cloner { t: &23 };
    let y = get(&x);
    assert_eq!(y, 23);
}
