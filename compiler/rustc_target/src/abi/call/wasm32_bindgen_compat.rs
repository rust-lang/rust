// This is not and has never been a correct C ABI for WebAssembly, but
// for a long time this was the C ABI that Rust used. wasm-bindgen
// depends on ABI details for this ABI and is incompatible with the
// correct C ABI, so this ABI is being kept around until wasm-bindgen
// can be fixed to work with the correct ABI. See #63649 for further
// discussion.

use crate::abi::call::{ArgAbi, FnAbi};

fn classify_ret<Ty>(ret: &mut ArgAbi<'_, Ty>) {
    ret.extend_integer_width_to(32);
}

fn classify_arg<Ty>(arg: &mut ArgAbi<'_, Ty>) {
    arg.extend_integer_width_to(32);
}

pub fn compute_abi_info<Ty>(fn_abi: &mut FnAbi<'_, Ty>) {
    if !fn_abi.ret.is_ignore() {
        classify_ret(&mut fn_abi.ret);
    }

    for arg in &mut fn_abi.args {
        if arg.is_ignore() {
            continue;
        }
        classify_arg(arg);
    }
}
