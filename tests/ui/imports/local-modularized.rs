// build-pass (FIXME(62277): could be check-pass?)

#[macro_export(local_inner_macros)]
macro_rules! dollar_crate_exported {
    (1) => { $crate::exported!(); };
    (2) => { exported!(); };
}

// Before `exported` is defined
exported!();

mod inner {

    ::exported!();
    crate::exported!();
    dollar_crate_exported!(1);
    dollar_crate_exported!(2);

    mod inner_inner {
        #[macro_export]
        macro_rules! exported {
            () => ()
        }
    }

    // After `exported` is defined
    ::exported!();
    crate::exported!();
    dollar_crate_exported!(1);
    dollar_crate_exported!(2);
}

exported!();

fn main() {}
