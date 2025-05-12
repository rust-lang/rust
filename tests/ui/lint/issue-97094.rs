#![deny(warnings)]

// Ensure that unknown lints inside cfg-attr's are linted for

#![cfg_attr(all(), allow(nonex_lint_top_level))]
//~^ ERROR unknown lint
#![cfg_attr(all(), allow(bare_trait_object))]
//~^ ERROR has been renamed

#[cfg_attr(all(), allow(nonex_lint_mod))]
//~^ ERROR unknown lint
mod baz {
    #![cfg_attr(all(), allow(nonex_lint_mod_inner))]
    //~^ ERROR unknown lint
}

#[cfg_attr(all(), allow(nonex_lint_fn))]
//~^ ERROR unknown lint
pub fn main() {}

macro_rules! bar {
    ($($t:tt)*) => {
        $($t)*
    };
}

bar!(
    #[cfg_attr(all(), allow(nonex_lint_in_macro))]
    //~^ ERROR unknown lint
    pub fn _bar() {}
);

// No warning for non-applying cfg
#[cfg_attr(any(), allow(nonex_lint_fn))]
pub fn _foo() {}

// Allowing unknown lints works if inside cfg_attr
#[cfg_attr(all(), allow(unknown_lints))]
mod bar_allowed {
    #[allow(nonex_lint_fn)]
    fn _foo() {}
}

// ... but not if the cfg_attr doesn't evaluate
#[cfg_attr(any(), allow(unknown_lints))]
mod bar_not_allowed {
    #[allow(nonex_lint_fn)]
    //~^ ERROR unknown lint
    fn _foo() {}
}
