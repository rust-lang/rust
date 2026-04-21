//@ proc-macro: has_helper_attr.rs
#![feature(custom_inner_attributes)]
#![feature(macro_attr)]
#![feature(stmt_expr_attributes)]
#![feature(const_block_items)]
#![feature(register_tool)]
#![register_tool(custom)]
#![feature(decl_macro)]

mod rustfmt_inner {
    #![rustfmt::skip]
}

mod rustfmt_ambig_unresolved {
    #![rustfmt::skip]
    //~^ ERROR cannot find `skip` in `rustfmt`
    //~| ERROR `rustfmt` is ambiguous

    mod rustfmt {}
}

// like above, but import exists
mod rustfmt_ambig {
    #![rustfmt::skip]
    //~^ ERROR `rustfmt` is ambiguous

    mod rustfmt {
        pub macro skip() {}
    }
}

mod rustfmt_inner_renamed {
    #![rust_fmt::skip]

    use rustfmt as rust_fmt;
    //~^ ERROR unresolved import `rustfmt`
}

#[custom::attr]
mod both_forms {
    #![custom::attr]

    #[custom::attr]
    mod both_forms_inner {
        #![custom::attr]
    }
}

extern crate has_helper_attr;

#[derive(has_helper_attr::has_helper_attr)]
struct InnerHelperUsed {
    #[helper]
    field: [(); {
        #[helper]
        mod inner {
            #![helper]
            //~^ ERROR cannot find attribute `helper` in this scope

            mod inner {
                #![helper]
                //~^ ERROR cannot find attribute `helper` in this scope
            }
        }

        #[helper_renamed]
        mod inner_2 {}

        use helper as helper_renamed;
        //~^ ERROR unresolved import `helper`

        0
    }],
}

mod rustfmt_ambig_via_macro {
    #![rustfmt::skip]
    //~^ ERROR cannot find `skip` in `rustfmt`
    //~| ERROR `rustfmt` is ambiguous

    macro_rules! define {
        () => { mod rustfmt {} };
    }

    define!();
}

// In these test cases, when we refer to a name that does not exist,
// we throw an error, even though if the name does exist, that attribute is
// NOT used. (inner attribute resolution happens from inside the module)

mod clippy {
    #[macro_export]
    macro_rules! example {
        attr() () => { compile_error!() };
    }
    pub use crate::example;
}

mod zx {
    #![clippy::example]
    #![clippy::non_existing]
}

trait Z {
    #![clippy::example]
    #![clippy::non_existing]
    //~^ ERROR cannot find `non_existing` in `clippy`
}

struct Zm;

impl Z for Zm {
    #![clippy::example]
    #![clippy::non_existing]
    //~^ ERROR cannot find `non_existing` in `clippy`
}

const {
    #![clippy::example]
    //~^ ERROR an inner attribute is not permitted in this context
}

const _: () = {
    #![clippy::example]
    #![clippy::non_existing]
};

mod zy {
    #![clippy]
    //~^ ERROR cannot find attribute `clippy` in this scope
}

fn main() {
    #![clippy::example]
    #![clippy::non_existing]
    //~^ ERROR cannot find `non_existing` in `clippy`
}
