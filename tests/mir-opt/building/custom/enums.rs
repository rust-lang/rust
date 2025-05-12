// skip-filecheck
#![feature(custom_mir, core_intrinsics)]

extern crate core;
use core::intrinsics::mir::*;

// EMIT_MIR enums.switch_bool.built.after.mir
#[custom_mir(dialect = "built")]
pub fn switch_bool(b: bool) -> u32 {
    mir! {
        {
            match b {
                true => t,
                false => f,
                _ => f,
            }
        }

        t = {
            RET = 5;
            Return()
        }

        f = {
            RET = 10;
            Return()
        }
    }
}

// EMIT_MIR enums.switch_option.built.after.mir
#[custom_mir(dialect = "built")]
pub fn switch_option(option: Option<()>) -> bool {
    mir! {
        {
            let discr = Discriminant(option);
            match discr {
                0 => n,
                1 => s,
                _ => s,
            }
        }

        n = {
            RET = false;
            Return()
        }

        s = {
            RET = true;
            Return()
        }
    }
}

#[repr(u8)]
enum Bool {
    False = 0,
    True = 1,
}

// EMIT_MIR enums.switch_option_repr.built.after.mir
#[custom_mir(dialect = "built")]
fn switch_option_repr(option: Bool) -> bool {
    mir! {
        {
            let discr = Discriminant(option);
            match discr {
                0 => f,
                _ => t,
            }
        }

        t = {
            RET = true;
            Return()
        }

        f = {
            RET = false;
            Return()
        }
    }
}

// EMIT_MIR enums.set_discr.built.after.mir
#[custom_mir(dialect = "runtime", phase = "initial")]
fn set_discr(option: &mut Option<()>) {
    mir! {
        {
            Deinit(*option);
            SetDiscriminant(*option, 0);
            Return()
        }
    }
}

// EMIT_MIR enums.set_discr_repr.built.after.mir
#[custom_mir(dialect = "runtime", phase = "initial")]
fn set_discr_repr(b: &mut Bool) {
    mir! {
        {
            SetDiscriminant(*b, 0);
            Return()
        }
    }
}

fn main() {
    assert_eq!(switch_bool(true), 5);
    assert_eq!(switch_bool(false), 10);

    assert_eq!(switch_option(Some(())), true);
    assert_eq!(switch_option(None), false);

    assert_eq!(switch_option_repr(Bool::True), true);
    assert_eq!(switch_option_repr(Bool::False), false);

    let mut opt = Some(());
    set_discr(&mut opt);
    assert_eq!(opt, None);

    let mut b = Bool::True;
    set_discr_repr(&mut b);
    assert!(matches!(b, Bool::False));
}
