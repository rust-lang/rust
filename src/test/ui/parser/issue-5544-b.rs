// compile-flags: -Z parse-only

fn main() {
    let __isize = 0xffff_ffff_ffff_ffff_ffff_ffff_ffff_ffff_ff;
    //~^ ERROR int literal is too large
}
