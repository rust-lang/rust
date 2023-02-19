#![feature(never_type)]
#![feature(exhaustive_patterns)]

mod private {
    pub struct Private {
        _bot: !,
        pub misc: bool,
    }
    pub const DATA: Option<Private> = None;
}

fn main() {
    match private::DATA {
    //~^ ERROR match is non-exhaustive
        None => {}
        Some(private::Private {
            misc: false,
            ..
        }) => {}
    }
}
