#![feature(plugin)]
#![feature(const_fn)]
#![plugin(clippy)]

#![deny(clippy_pedantic)]
#![allow(unused, missing_docs_in_private_items)]

fn do_nothing<T>(_: T) {}

fn diverge<T>(_: T) -> ! {
    panic!()
}

fn plus_one(value: usize) -> usize {
    value + 1
}

struct HasOption {
    field: Option<usize>,
}

impl HasOption {
    fn do_option_nothing(self: &HasOption, value: usize) {}

    fn do_option_plus_one(self: &HasOption, value: usize) -> usize {
        value + 1
    }
}

#[cfg_attr(rustfmt, rustfmt_skip)]
fn main() {
    let x = HasOption { field: Some(10) };

    x.field.map(plus_one);
    let _ : Option<()> = x.field.map(do_nothing);

    x.field.map(do_nothing);
    //~^ ERROR called `map(f)` on an Option value where `f` is a nil function
    //~| HELP try this
    //~| SUGGESTION if let Some(...) = x.field { do_nothing(...) }

    x.field.map(do_nothing);
    //~^ ERROR called `map(f)` on an Option value where `f` is a nil function
    //~| HELP try this
    //~| SUGGESTION if let Some(...) = x.field { do_nothing(...) }

    x.field.map(diverge);
    //~^ ERROR called `map(f)` on an Option value where `f` is a nil function
    //~| HELP try this
    //~| SUGGESTION if let Some(...) = x.field { diverge(...) }
}
