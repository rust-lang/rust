//@ run-pass
#[derive(PartialEq, Debug)]
pub struct Partial<T> { x: T, y: T }

#[derive(PartialEq, Debug)]
struct S { val: isize }
impl S { fn new(v: isize) -> S { S { val: v } } }
impl Drop for S { fn drop(&mut self) { } }

pub type Two<T> = (Partial<T>, Partial<T>);

pub fn f<T, F>((b1, b2): (T, T), (b3, b4): (T, T), mut f: F) -> Two<T> where F: FnMut(T) -> T {
    let p = Partial { x: b1, y: b2 };
    let q = Partial { x: b3, y: b4 };

     // Move of `q` is legal even though we have already moved `q.y`;
     // the `..q` moves all fields *except* `q.y` in this context.
     // Likewise, the move of `p.x` is legal for similar reasons.
    (Partial { x: f(q.y), ..p }, Partial { y: f(p.x), ..q })
}

pub fn main() {
    let two = f((S::new(1), S::new(3)),
                (S::new(5), S::new(7)),
                |S { val: z }| S::new(z+1));
    assert_eq!(two, (Partial { x: S::new(8), y: S::new(3) },
                     Partial { x: S::new(5), y: S::new(2) }));
}
