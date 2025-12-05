#[repr(u8)]
enum Eu8 {
    Au8 = 23,
    Bu8 = 223,
    Cu8 = -23,
    //~^ ERROR cannot apply unary operator `-` to type `u8`
}

#[repr(u16)]
enum Eu16 {
    Au16 = 23,
    Bu16 = 55555,
    Cu16 = -22333,
    //~^ ERROR cannot apply unary operator `-` to type `u16`
}

#[repr(u32)]
enum Eu32 {
    Au32 = 23,
    Bu32 = 3_000_000_000,
    Cu32 = -2_000_000_000,
    //~^ ERROR cannot apply unary operator `-` to type `u32`
}

#[repr(u64)]
enum Eu64 {
    Au32 = 23,
    Bu32 = 3_000_000_000,
    Cu32 = -2_000_000_000,
    //~^ ERROR cannot apply unary operator `-` to type `u64`
}

// u64 currently allows negative numbers, and i64 allows numbers greater than `1<<63`.  This is a
// little counterintuitive, but since the discriminant can store all the bits, and extracting it
// with a cast requires specifying the signedness, there is no loss of information in those cases.
// This also applies to isize and usize on 64-bit targets.

pub fn main() { }
