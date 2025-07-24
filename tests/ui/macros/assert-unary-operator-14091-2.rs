//

// Very

// sensitive
pub struct BytePos(pub u32);

// to particular

// line numberings / offsets

fn main() {
    let x = BytePos(1);

    assert!(x, x);
    //~^ ERROR cannot apply unary operator `!` to type `BytePos`
}
