//@ aux-crate:reexport_self_dep=reexport_self_dep.rs
//@ edition: 2021

#![crate_name = "foo"]

// Case 1: Standard `{self}`
pub mod case1 {
    //@ has 'foo/case1/index.html' '//*[@class="item-table reexports"]' 'pub use reexport_self_dep;'
    //@ !has 'foo/case1/index.html' '//code' 'pub use self;'

    pub use reexport_self_dep::{self};
}

// Case 2: Renamed `{self as alias}`
pub mod case2 {
    //@ has 'foo/case2/index.html' '//*[@class="item-table reexports"]' 'pub use reexport_self_dep as my_dep;'
    //@ !has 'foo/case2/index.html' '//code' 'pub use self as my_dep;'

    pub use reexport_self_dep::{self as my_dep};
}

// Case 3: Mixed import with both `{self}` and another item
pub mod case3 {
    //@ has 'foo/case3/index.html' '//*[@class="item-table reexports"]' 'pub use reexport_self_dep;'
    //@ !has 'foo/case3/index.html' '//code' 'pub use self;'
    //@ has 'foo/case3/fn.some_item.html'

    pub use reexport_self_dep::{self, some_item};
}
