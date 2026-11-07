//@ edition: 2024
//@ check-pass

// In edition 2024, the anonymous lifetime in `&self` is captured by the opaque
// type. However, because it is unrelated to `'a` (which is the outlives bound
// on the `impl Trait`), we know that the hidden type cannot contain that
// anonymous lifetime and we should not treat it as live within the opaque type.

#![forbid(unsafe_code)]

struct Guard<T, F> {
    val: T,
    f: F,
}

struct Table {
    items: usize,
}

impl Table {
    fn prepare<'a, A>(&self, alloc: &'a A) -> Guard<usize, impl FnMut(&mut usize) + 'a> {
        let _ = alloc;
        Guard { val: self.items, f: |_: &mut usize| {} }
    }

    fn resize<A>(&mut self, alloc: &A) {
        let mut g = self.prepare(alloc);
        self.items += 1;
        g.val += 1;
        (g.f)(&mut g.val);
    }
}

fn main() {}
