//@ edition:2021

// Don't ICE in ByMove shim builder when MIR body is tainted by writeback errors

fn main() {
    let _ = async || {
        used_fn();
        //~^ ERROR cannot find function `used_fn` in this scope
        0
    };
}
