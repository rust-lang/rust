//@ pp-exact

const C_PTR: () = { let a = 1; &raw const a; };
static S_PTR: () = { let b = false; &raw const b; };

fn main() {
    let x = 123;
    let mut y = 345;
    let c_p = &raw const x;
    let parens = unsafe { *(&raw mut (y)) };
}
