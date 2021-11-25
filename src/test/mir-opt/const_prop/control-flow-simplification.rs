// compile-flags: -Zmir-opt-level=1

trait HasSize: Sized {
    const BYTES:usize = std::mem::size_of::<Self>();
}

impl<This> HasSize for This{}

// EMIT_MIR control_flow_simplification.hello.ConstProp.diff
// EMIT_MIR control_flow_simplification.hello.PreCodegen.before.mir
fn hello<T>(){
    if <bool>::BYTES > 10 {
        panic!()
    }
}

pub fn main() {
    hello::<()>();
    hello::<Vec<()>>();
}
