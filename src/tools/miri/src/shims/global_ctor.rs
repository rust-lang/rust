//! Implement global constructors.

use std::task::Poll;

use rustc_abi::ExternAbi;
use rustc_target::spec::BinaryFormat;

use crate::*;

#[derive(Debug, Default)]
pub struct GlobalCtorState<'tcx>(GlobalCtorStatePriv<'tcx>);

#[derive(Debug, Default)]
enum GlobalCtorStatePriv<'tcx> {
    #[default]
    Init,
    /// The list of constructor functions that we still have to call.
    Ctors(Vec<ImmTy<'tcx>>),
    Done,
}

impl<'tcx> GlobalCtorState<'tcx> {
    pub fn on_stack_empty(
        &mut self,
        this: &mut MiriInterpCx<'tcx>,
    ) -> InterpResult<'tcx, Poll<()>> {
        use GlobalCtorStatePriv::*;
        let new_state = 'new_state: {
            match &mut self.0 {
                Init => {
                    let this = this.eval_context_mut();

                    // Lookup constructors from the relevant magic link section.
                    let ctors = match this.tcx.sess.target.binary_format {
                        // Read the CRT library section on Windows.
                        BinaryFormat::Coff =>
                            this.lookup_link_section(|section| section == ".CRT$XCU")?,

                        // Read the `__mod_init_func` section on macOS.
                        BinaryFormat::MachO =>
                            this.lookup_link_section(|section| {
                                let mut parts = section.splitn(3, ',');
                                let (segment_name, section_name, section_type) =
                                    (parts.next(), parts.next(), parts.next());

                                segment_name == Some("__DATA")
                                    && section_name == Some("__mod_init_func")
                                    // The `mod_init_funcs` directive ensures that the
                                    // `S_MOD_INIT_FUNC_POINTERS` flag is set on the section. LLVM
                                    // adds this automatically so we currently do not require it.
                                    // FIXME: is this guaranteed LLVM behavior? If not, we shouldn't
                                    // implicitly add it here. Also see
                                    // <https://github.com/rust-lang/miri/pull/4459#discussion_r2200115629>.
                                    && matches!(section_type, None | Some("mod_init_funcs"))
                            })?,

                        // Read the standard `.init_array` section on platforms that use ELF, or WASM,
                        // which supports the same linker directive.
                        // FIXME: Add support for `.init_array.N` and `.ctors`?
                        BinaryFormat::Elf | BinaryFormat::Wasm =>
                            this.lookup_link_section(|section| section == ".init_array")?,

                        // Other platforms have no global ctor support.
                        _ => break 'new_state Done,
                    };

                    break 'new_state Ctors(ctors);
                }
                Ctors(ctors) => {
                    if let Some(ctor) = ctors.pop() {
                        let this = this.eval_context_mut();

                        let ctor = ctor.to_scalar().to_pointer(this)?;
                        let thread_callback = this.get_ptr_fn(ctor)?.as_instance()?;

                        // The signature of this function is `unsafe extern "C" fn()`.
                        this.call_function(
                            thread_callback,
                            ExternAbi::C { unwind: false },
                            &[],
                            None,
                            ReturnContinuation::Stop { cleanup: true },
                        )?;

                        return interp_ok(Poll::Pending); // we stay in this state (but `ctors` got shorter)
                    }

                    // No more constructors to run.
                    break 'new_state Done;
                }
                Done => return interp_ok(Poll::Ready(())),
            }
        };

        self.0 = new_state;
        interp_ok(Poll::Pending)
    }
}
