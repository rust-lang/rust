use rustc::mir;

use crate::*;

#[derive(Debug, Copy, Clone)]
pub enum Dlsym {
    GetEntropy,
}

impl Dlsym {
    pub fn from_str(name: &str) -> Option<Dlsym> {
        use self::Dlsym::*;
        Some(match name {
            "getentropy" => GetEntropy,
            _ => return None,
        })
    }
}

impl<'mir, 'tcx> EvalContextExt<'mir, 'tcx> for crate::MiriEvalContext<'mir, 'tcx> {}
pub trait EvalContextExt<'mir, 'tcx: 'mir>: crate::MiriEvalContextExt<'mir, 'tcx> {
    fn call_dlsym(
        &mut self,
        dlsym: Dlsym,
        args: &[OpTy<'tcx, Tag>],
        dest: Option<PlaceTy<'tcx, Tag>>,
        ret: Option<mir::BasicBlock>,
    ) -> InterpResult<'tcx> {
        use self::Dlsym::*;

        let this = self.eval_context_mut();

        let dest = dest.expect("we don't support any diverging dlsym");
        let ret = ret.expect("dest is `Some` but ret is `None`");
        
        match dlsym {
            GetEntropy => {
                let ptr = this.read_scalar(args[0])?.not_undef()?;
                let len = this.read_scalar(args[1])?.to_usize(this)?;
                this.gen_random(len as usize, ptr)?;
                this.write_null(dest)?;
            }
        }

        this.goto_block(Some(ret))?;
        this.dump_place(*dest);
        Ok(())
    }
}
