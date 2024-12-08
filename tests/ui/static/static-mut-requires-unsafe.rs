static mut a: isize = 3;

fn main() {
    a += 3;         //~ ERROR: requires unsafe
    a = 4;          //~ ERROR: requires unsafe
    let _b = a;     //~ ERROR: requires unsafe
}
