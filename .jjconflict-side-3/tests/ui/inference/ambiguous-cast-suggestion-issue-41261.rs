//@ edition: 2021

// Regression test for issue #41261.
// The diagnostic should point at the ambiguous cast, not at the later method call.

struct S {
    v: Vec<(u32, Vec<u32>)>,
}

impl S {
    pub fn remove(&mut self, i: u32) -> Option<std::vec::Drain<'_, u32>> {
        self.v.get_mut(i as _).map(|&mut (_, ref mut v2)| {
            //~^ ERROR type annotations needed
            v2.drain(..)
        })
    }
}

fn main() {}
