// compile-flags: -Zmir-opt-level=1

trait NeedsDrop:Sized{
    const NEEDS:bool=std::mem::needs_drop::<Self>();
}

impl<This> NeedsDrop for This{}

// EMIT_MIR rustc.hello.ConstProp.diff
// EMIT_MIR rustc.hello.PreCodegen.before.mir
fn hello<T>(){
    if <bool>::NEEDS {
        panic!()
    }
}

pub fn main() {
    hello::<()>();
    hello::<Vec<()>>();
}
