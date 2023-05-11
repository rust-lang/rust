// compile-flags: -Zunstable-options

pub fn issue_111280() {
    struct_span_err(msg).emit(); //~ ERROR cannot find value `msg`
    //~^ ERROR cannot find function `struct_span_err`
}

fn main() {}
