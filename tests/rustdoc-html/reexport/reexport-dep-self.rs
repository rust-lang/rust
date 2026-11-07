//@ aux-build: reexport-dep-self.rs

#![crate_name = "foo"]

extern crate reexport_dep_self;

pub mod extern_crate_self {
    //@ has 'foo/extern_crate_self/index.html' '//*[@class="item-table reexports"]' \
    // 'pub use reexport_dep_self;'
    pub use reexport_dep_self::{self};
}

pub mod extern_crate_self_rename {
    //@ has 'foo/extern_crate_self_rename/index.html' '//*[@class="item-table reexports"]' \
    // 'pub use reexport_dep_self as my_dep;'
    pub use reexport_dep_self::{self as my_dep};
}

pub mod extern_crate_self_multi {
    //@ has 'foo/extern_crate_self_multi/index.html' '//*[@class="item-table reexports"]' \
    // 'pub use reexport_dep_self;'
    pub use reexport_dep_self::{self, foo};
}
