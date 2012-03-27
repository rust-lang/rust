// error-pattern:invalidate reference x

fn whoknows(x: @mut {mut x: int}) { x.x = 10; }

fn main() {
    let box = @mut {mut x: 1};
    alt *box { x { whoknows(box); log(error, x); } }
}
