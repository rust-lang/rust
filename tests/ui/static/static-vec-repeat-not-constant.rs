fn foo() -> isize {
    23
}

static A: [isize; 2] = [foo(); 2];
//~^ ERROR: E0015

fn main() {}
