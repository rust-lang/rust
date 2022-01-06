// Regression test for issue 81708 and issue 91816 where running a drop
// elaboration on a MIR which failed borrowck lead to an ICE.

static A: () = {
    let a: [String; 1];
    //~^ ERROR destructors cannot be evaluated at compile-time
    a[0] = String::new();
    //~^ ERROR destructors cannot be evaluated at compile-time
    //~| ERROR use of possibly-uninitialized variable
};

struct B<T>([T; 1]);

impl<T> B<T> {
    pub const fn f(mut self, other: T) -> Self {
        let _this = self;
        //~^ ERROR destructors cannot be evaluated at compile-time
        self.0[0] = other;
        //~^ ERROR destructors cannot be evaluated at compile-time
        //~| ERROR use of moved value
        self
    }
}

fn main() {}
