// revisions: min_exhaustive_patterns exhaustive_patterns
#![cfg_attr(exhaustive_patterns, feature(exhaustive_patterns))]
#![cfg_attr(min_exhaustive_patterns, feature(min_exhaustive_patterns))]
//[min_exhaustive_patterns]~^ WARN the feature `min_exhaustive_patterns` is incomplete
#![feature(never_type)]

mod private {
    pub struct Private {
        _bot: !,
        pub misc: bool,
    }
    pub const DATA: Option<Private> = None;
}

fn main() {
    match private::DATA {
        //~^ ERROR non-exhaustive patterns: `Some(Private { misc: true, .. })` not covered
        None => {}
        Some(private::Private { misc: false, .. }) => {}
    }
}
