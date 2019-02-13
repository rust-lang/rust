use crate::abi::call::{ArgType, FnType, Reg};
use crate::abi::Abi;

// Win64 ABI: http://msdn.microsoft.com/en-us/library/zthk2dkh.aspx

pub fn compute_abi_info<Ty>(fty: &mut FnType<'_, Ty>) {
    let fixup = |a: &mut ArgType<'_, Ty>| {
        match a.layout.abi {
            Abi::Uninhabited => {}
            Abi::ScalarPair(..) |
            Abi::Aggregate { .. } => {
                match a.layout.size.bits() {
                    8 => a.cast_to(Reg::i8()),
                    16 => a.cast_to(Reg::i16()),
                    32 => a.cast_to(Reg::i32()),
                    64 => a.cast_to(Reg::i64()),
                    _ => a.make_indirect()
                }
            }
            Abi::Vector { .. } => {
                // FIXME(eddyb) there should be a size cap here
                // (probably what clang calls "illegal vectors").
            }
            Abi::Scalar(_) => {
                if a.layout.size.bytes() > 8 {
                    a.make_indirect();
                } else {
                    a.extend_integer_width_to(32);
                }
            }
        }
    };

    if !fty.ret.is_ignore() {
        fixup(&mut fty.ret);
    }
    for arg in &mut fty.args {
        if arg.is_ignore() { continue; }
        fixup(arg);
    }
}
