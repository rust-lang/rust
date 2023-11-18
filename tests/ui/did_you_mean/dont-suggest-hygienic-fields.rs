// Regression test for issue #116334.
// Don't include hygienic fields from different syntax contexts in
// the list of available or similarly named fields.

#![feature(decl_macro)]

//
// Test cases where we should *not* suggest hygienic fields since
// they are inaccessible in the respective syntax context:
//

macro compound($Ty:ident) {
    #[derive(Default)]
    struct $Ty {
        field: u32, // field `field` is hygienic
    }
}

macro component($Ty:ident) {
    struct $Ty(u64); // field `0` is hygienic (but still accessible via the constructor)
}

compound! { Compound }
component! { Component }

macro inject_into_expr_bad($field:ident) {{
    struct Casket {
        field: (),
    }

    const CASKET: Casket = Casket { field: () };

    CASKET.$field
}}

macro inject_into_pat_bad($( $any:tt )*) {{
    struct Casket { field: () }

    const CASKET: Casket = Casket { field: () };

    let Casket { $( $any )* } = CASKET; //~ ERROR pattern does not mention field `field`
}}

fn main() {
    let ty = Compound::default();

    let _ = ty.field; //~ ERROR no field `field` on type `Compound`
    let _ = ty.fieeld; //~ ERROR no field `fieeld` on type `Compound`

    let Compound { field } = ty;
    //~^ ERROR struct `Compound` does not have a field named `field`
    //~| ERROR pattern requires `..` due to inaccessible fields
    //~| HELP ignore the inaccessible and unused fields

    let ty = Component(90);

    let _ = ty.0; //~ ERROR no field `0` on type `Component`

    let _ = inject_into_expr_bad!(field); //~ ERROR no field `field` on type `main::Casket`

    inject_into_pat_bad!(field: _); //~ ERROR struct `main::Casket` does not have a field named `field`
}

//
// Test cases where we *should* suggest hygienic fields since
// they're accessible in the respective syntax context:
//

environment!();

struct Case { field: () }

macro environment() {
    struct Crate { field: () }

    // Here, we do want to suggest `field` even though it's hygienic
    // precisely because they come from the same syntax context.
    const CRATE: Crate = Crate { fiel: () };
    //~^ ERROR struct `Crate` has no field named `fiel`
    //~| HELP a field with a similar name exists

    // `field` isn't in the same syntax context but in a parent one
    // which means it's accessible and should be suggested.
    const CASE: Case = Case { fiel: () };
    //~^ ERROR struct `Case` has no field named `fiel`
    //~| HELP a field with a similar name exists
}

macro inject_into_expr_good($field_def_site:ident, $field_use_site:ident) {{
    struct Carton {
        $field_def_site: (),
    }

    const CARTON: Carton = Carton { $field_def_site: () };

    CARTON.$field_use_site
}}

fn scope() {
    // It's immaterial that `CARTON` / the field expression as a whole isn't accessible in this
    // syntax context, the identifier substituted for `$field_use_site` *is*.
    let _ = inject_into_expr_good!(field, fiel);
    //~^ ERROR no field `fiel` on type `Carton`
    //~| HELP a field with a similar name exists
}
