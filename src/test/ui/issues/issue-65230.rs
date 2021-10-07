trait T {
    type U;
    fn f(&self) -> Self::U;
}

struct X<'a>(&'a mut i32);

impl<'a> T for X<'a> {
    type U = &'a i32;
    fn f(&self) -> Self::U {
        self.0
    }
    //~^^^ ERROR cannot infer an appropriate lifetime for lifetime parameter `'a`
    //
    // Return type of `f` has lifetime `'a` but it tries to return `self.0` which
    // has lifetime `'_`.
}

fn main() {}
