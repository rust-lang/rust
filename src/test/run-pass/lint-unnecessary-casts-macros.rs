
#[feature(macro_rules)];

macro_rules! macro_with_cast(
    ($x:ident) => ( // invoke it like `(input_5 special_e)`
        { let y: u32 = 1;
        foo(y);
        $x as u64
    });
)

fn foo(z: u32) -> u32 { z+z }

fn main() {
    let x: u64 = 10;
    macro_with_cast!(x);
    macro_with_cast!(x);
    //println!("{}", macro_with_cast!(x));
}
