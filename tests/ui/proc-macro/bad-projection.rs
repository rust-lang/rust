// force-host
// no-prefer-dynamic

#![crate_type = "proc-macro"]
#![allow(warnings)]

extern crate proc_macro;

trait Project {
    type Assoc;
}

#[proc_macro]
pub fn uwu() -> <() as Project>::Assoc {}
//~^ ERROR the trait bound `(): Project` is not satisfied
