// Regression test for issue 81708 and issue 91816 where running a drop
// elaboration on a MIR which failed borrowck lead to an ICE.

static A: () = {
    let a: [String; 1];
    //~^ ERROR destructor of
    a[0] = String::new();
    //~^ ERROR destructor of
};

struct B<T>([T; 1]);

impl<T> B<T> {
    pub const fn f(mut self, other: T) -> Self {
        let _this = self;
        //~^ ERROR destructor of
        self.0[0] = other;
        //~^ ERROR destructor of
        self
    }
}

fn main() {}
