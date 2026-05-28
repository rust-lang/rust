fn foo() -> isize { 23 }

static a: [isize; 2] = [foo(); 2];
//~^ ERROR: E0015

fn main() {}
