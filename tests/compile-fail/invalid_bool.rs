//ignore-test FIXME (do some basic validation of invariants for all values in flight)
//This does currently not get caught becuase it compiles to SwitchInt, which
//has no knowledge about data invariants.

fn main() {
    let b = unsafe { std::mem::transmute::<u8, bool>(2) };
    if b { unreachable!() } else { unreachable!() } //~ ERROR constant evaluation error
    //~^ NOTE invalid boolean value read
}
