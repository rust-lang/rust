fn foo() -> isize { 23 }

static a: [isize; 2] = [foo(); 2];
//~^ ERROR: cannot call non-const fn `foo` in statics [E0015]

fn main() {}
