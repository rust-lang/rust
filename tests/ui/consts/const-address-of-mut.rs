//@check-pass

const A: () = { let mut x = 2; &raw mut x; };

static B: () = { let mut x = 2; &raw mut x; };

const fn foo() {
    let mut x = 0;
    let y = &raw mut x;
}

fn main() {}
