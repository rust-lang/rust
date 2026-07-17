//@ aux-crate:priv:macro_provenance_leaf=macro_provenance_leaf.rs
//@ aux-crate:macro_provenance_facade=macro_provenance_facade.rs
//@ proc-macro:priv:macro_provenance_pm.rs
//@ compile-flags: -Zunstable-options

#![crate_type = "lib"]
#![deny(exported_private_dependencies)]

extern crate macro_provenance_facade as public_macros;
extern crate macro_provenance_leaf as private_macros;
extern crate macro_provenance_pm;

pub use private_macros::definition_side as private_reexport;
//~^ ERROR macro `private_reexport` from private dependency 'macro_provenance_leaf' is re-exported

pub use public_macros::definition_side as public_reexport;

private_macros::definition_side!(private_definition_side);
//~^ ERROR type `Hidden` from private dependency 'macro_provenance_leaf' in public interface

public_macros::definition_side!(public_definition_side);

public_macros::captured_type!(captured_private, private_macros::Hidden);
//~^ ERROR type `Hidden` from private dependency 'macro_provenance_leaf' in public interface

public_macros::captured_type!(captured_public, public_macros::Hidden);

private_macros::call_site_core!(call_site_core);

public_macros::call_site_private!(call_site_private);
//~^ ERROR type `Hidden` from private dependency 'macro_provenance_leaf' in public interface

private_macros::nested!(private_nested);
//~^ ERROR type `Hidden` from private dependency 'macro_provenance_leaf' in public interface

public_macros::nested!(public_nested);

macro_provenance_pm::call_site_path!();
//~^ ERROR type `Hidden` from private dependency 'macro_provenance_leaf' in public interface
