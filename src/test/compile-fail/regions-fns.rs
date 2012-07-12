// Should fail region checking, because g can only accept a pointer
// with lifetime r, and a is a pointer with unspecified lifetime.
fn not_ok_1(a: &uint) {
    let mut g: fn@(x: &uint) = fn@(x: &r/uint) {};
    //~^ ERROR mismatched types
    g(a);
}

// Should fail region checking, because g can only accept a pointer
// with lifetime r, and a is a pointer with lifetime s.
fn not_ok_2(s: &s/uint)
{
    let mut g: fn@(x: &uint) = fn@(x: &r/uint) {};
    //~^ ERROR mismatched types
    g(s);
}

fn main() {
}


