use rustc_middle::mir;

use crate::helpers::check_arg_count;
use crate::*;

#[derive(Debug, Copy, Clone)]
#[allow(non_camel_case_types)]
pub enum Dlsym {
    signal,
}

impl Dlsym {
    // Returns an error for unsupported symbols, and None if this symbol
    // should become a NULL pointer (pretend it does not exist).
    pub fn from_str<'tcx>(name: &str) -> InterpResult<'tcx, Option<Dlsym>> {
        Ok(match name {
            "signal" => Some(Dlsym::signal),
            "android_set_abort_message" => None,
            _ => throw_unsup_format!("unsupported Android dlsym: {}", name),
        })
    }
}

impl<'mir, 'tcx: 'mir> EvalContextExt<'mir, 'tcx> for crate::MiriInterpCx<'mir, 'tcx> {}
pub trait EvalContextExt<'mir, 'tcx: 'mir>: crate::MiriInterpCxExt<'mir, 'tcx> {
    fn call_dlsym(
        &mut self,
        dlsym: Dlsym,
        args: &[OpTy<'tcx, Provenance>],
        dest: &PlaceTy<'tcx, Provenance>,
        ret: Option<mir::BasicBlock>,
    ) -> InterpResult<'tcx> {
        let this = self.eval_context_mut();
        let ret = ret.expect("we don't support any diverging dlsym");
        assert!(this.tcx.sess.target.os == "android");

        match dlsym {
            Dlsym::signal => {
                if !this.frame_in_std() {
                    throw_unsup_format!(
                        "`signal` support is crude and just enough for libstd to work"
                    );
                }

                let &[ref _sig, ref _func] = check_arg_count(args)?;
                this.write_null(dest)?;
            }
        }

        log::trace!("{:?}", this.dump_place(**dest));
        this.go_to_block(ret);
        Ok(())
    }
}
