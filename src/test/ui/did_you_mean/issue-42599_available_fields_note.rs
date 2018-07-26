// Copyright 2017 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

mod submodule {

    #[derive(Default)]
    pub struct Demo {
        pub favorite_integer: isize,
        secret_integer: isize,
        pub innocently_misspellable: (),
        another_field: bool,
        yet_another_field: bool,
        always_more_fields: bool,
        and_ever: bool,
    }

    impl Demo {
        fn new_with_secret_two() -> Self {
            Self { secret_integer: 2, inocently_mispellable: () }
            //~^ ERROR no field
        }

        fn new_with_secret_three() -> Self {
            Self { secret_integer: 3, egregiously_nonexistent_field: () }
            //~^ ERROR no field
        }
    }

}

fn main() {
    use submodule::Demo;

    let demo = Demo::default();
    let innocent_field_misaccess = demo.inocently_mispellable;
    //~^ ERROR no field
    // note shouldn't suggest private fields
    let egregious_field_misaccess = demo.egregiously_nonexistent_field;
    //~^ ERROR no field
}
