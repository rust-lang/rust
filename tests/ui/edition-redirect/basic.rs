//@ revisions: edition2018 edition2021 edition2024
//@[edition2018] edition: 2018
//@[edition2021] edition: 2021
//@[edition2024] edition: 2024
//@ aux-build: basic.rs
//@ check-pass

#[macro_use]
extern crate basic as edition_redirect;

use edition_redirect::{
    Redirected as ImportedRedirected, redirected_macro as imported_redirected_macro,
};
use edition_redirect::{
    reexport_scope::Current as ExpectedScopedRedirected,
    use_targets::CurrentUse as ExpectedReexportedUse,
};

#[cfg(edition2018)]
use edition_redirect::{
    Oldest as ExpectedRedirected, use_targets::OldestUse as ExpectedRedirectedUse,
};
#[cfg(edition2021)]
use edition_redirect::{
    Middle as ExpectedRedirected, use_targets::MiddleUse as ExpectedRedirectedUse,
};
#[cfg(edition2024)]
use edition_redirect::{
    Redirected as ExpectedRedirected, use_targets::CurrentUse as ExpectedRedirectedUse,
};

#[cfg(edition2018)]
const EXPECTED_VALUE: usize = 1;
#[cfg(edition2021)]
const EXPECTED_VALUE: usize = 2;
#[cfg(edition2024)]
const EXPECTED_VALUE: usize = 3;

fn explicit() {
    let _: ExpectedRedirected = edition_redirect::Redirected;
    let _: ExpectedRedirectedUse = edition_redirect::RedirectedUse;
    let _: ExpectedScopedRedirected = edition_redirect::ScopedRedirected;
    let _: edition_redirect::same_redirects::Item = ExpectedReexportedUse;
    const _: [(); EXPECTED_VALUE] = [(); edition_redirect::redirected_module::VALUE];
    const _: [(); EXPECTED_VALUE] = [(); edition_redirect::redirected_macro!()];
    const _: [(); EXPECTED_VALUE] = [(); redirected_macro!()];
    let _: ImportedRedirected = ExpectedRedirected;
    const _: [(); EXPECTED_VALUE] = [(); imported_redirected_macro!()];
}

mod glob {
    use super::{EXPECTED_VALUE, ExpectedRedirected, ExpectedRedirectedUse};
    use edition_redirect::*;

    fn check() {
        let _: Redirected = ExpectedRedirected;
        let _: RedirectedUse = ExpectedRedirectedUse;
        const _: [(); EXPECTED_VALUE] = [(); redirected_module::VALUE];
        const _: [(); EXPECTED_VALUE] = [(); redirected_macro!()];
    }
}

fn main() {
    explicit();
}
