// only-x86_64

const HUGE_SIZE: usize = !0usize / 8;


pub struct TooBigArray {
    arr: [u8; HUGE_SIZE],
}

impl TooBigArray {
    pub const fn new() -> Self {
        TooBigArray { arr: [0x00; HUGE_SIZE], }
    }
}

static MY_TOO_BIG_ARRAY_1: TooBigArray = TooBigArray::new();
//~^ ERROR values of the type `[u8; 2305843009213693951]` are too big
static MY_TOO_BIG_ARRAY_2: [u8; HUGE_SIZE] = [0x00; HUGE_SIZE];
//~^ ERROR values of the type `[u8; 2305843009213693951]` are too big

fn main() { }
