// Enough cross-function structure that the partitioner spreads user code and
// exercises promotion: a shared static referenced from several functions and
// trait objects whose vtables are address-significant.

static TABLE: [u32; 8] = [3, 1, 4, 1, 5, 9, 2, 6];

#[used]
static KEEP: fn(u32) -> u32 = kept;

#[inline(never)]
fn kept(x: u32) -> u32 {
    x.wrapping_mul(7)
}

trait Op {
    fn apply(&self, x: u32) -> u32;
}

struct Add(u32);
struct Xor(u32);

impl Op for Add {
    fn apply(&self, x: u32) -> u32 {
        x.wrapping_add(self.0).wrapping_add(TABLE[(x % 8) as usize])
    }
}

impl Op for Xor {
    fn apply(&self, x: u32) -> u32 {
        (x ^ self.0).rotate_left(TABLE[(x % 8) as usize])
    }
}

#[inline(never)]
fn mix(ops: &[Box<dyn Op>], seed: u32) -> u32 {
    ops.iter().fold(seed, |acc, op| op.apply(acc))
}

fn main() {
    let ops: Vec<Box<dyn Op>> = vec![Box::new(Add(17)), Box::new(Xor(0x5a5a)), Box::new(Add(3))];
    println!("{}", mix(&ops, kept(41)));
}
