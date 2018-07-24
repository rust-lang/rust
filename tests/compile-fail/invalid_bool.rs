//ignore-test FIXME (do some basic validation of invariants for all values in flight)

fn main() {
    let b = unsafe { std::mem::transmute::<u8, bool>(2) };
    if b { unreachable!() } else { unreachable!() } //~ ERROR constant evaluation error
    //~^ NOTE invalid boolean value read
}
