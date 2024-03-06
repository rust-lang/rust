//@ aux-build:issue-59764.rs
//@ compile-flags:--extern issue_59764
//@ edition:2018

#![allow(warnings)]

// This tests the suggestion to import macros from the root of a crate. This aims to capture
// the case where a user attempts to import a macro from the definition location instead of the
// root of the crate and the macro is annotated with `#![macro_export]`.

// Edge cases..

mod multiple_imports_same_line_at_end {
    use issue_59764::foo::{baz, makro};
    //~^ ERROR unresolved import `issue_59764::foo::makro` [E0432]
}

mod multiple_imports_multiline_at_end_trailing_comma {
    use issue_59764::foo::{
        baz,
        makro, //~ ERROR unresolved import `issue_59764::foo::makro` [E0432]
    };
}

mod multiple_imports_multiline_at_end {
    use issue_59764::foo::{
        baz,
        makro //~ ERROR unresolved import `issue_59764::foo::makro` [E0432]
    };
}

mod multiple_imports_same_line_in_middle {
    use issue_59764::foo::{baz, makro, foobar};
    //~^ ERROR unresolved import `issue_59764::foo::makro` [E0432]
}

mod multiple_imports_multiline_in_middle_trailing_comma {
    use issue_59764::foo::{
        baz,
        makro, //~ ERROR unresolved import `issue_59764::foo::makro` [E0432]
        foobar,
    };
}

mod multiple_imports_multiline_in_middle {
    use issue_59764::foo::{
        baz,
        makro, //~ ERROR unresolved import `issue_59764::foo::makro` [E0432]
        foobar
    };
}

mod nested_imports {
    use issue_59764::{foobaz, foo::makro};
    //~^ ERROR unresolved import `issue_59764::foo::makro` [E0432]
}

mod nested_multiple_imports {
    use issue_59764::{foobaz, foo::{baz, makro}};
    //~^ ERROR unresolved import `issue_59764::foo::makro` [E0432]
}

mod nested_multiline_multiple_imports_trailing_comma {
    use issue_59764::{
        foobaz,
        foo::{
            baz,
            makro, //~ ERROR unresolved import `issue_59764::foo::makro` [E0432]
        },
    };
}

mod nested_multiline_multiple_imports {
    use issue_59764::{
        foobaz,
        foo::{
            baz,
            makro //~ ERROR unresolved import `issue_59764::foo::makro` [E0432]
        }
    };
}

mod doubly_nested_multiple_imports {
    use issue_59764::{foobaz, foo::{baz, makro, barbaz::{barfoo}}};
    //~^ ERROR unresolved import `issue_59764::foo::makro` [E0432]
}

mod doubly_multiline_nested_multiple_imports {
    use issue_59764::{
        foobaz,
        foo::{
            baz,
            makro, //~ ERROR unresolved import `issue_59764::foo::makro` [E0432]
            barbaz::{
                barfoo,
            }
        }
    };
}

mod renamed_import {
    use issue_59764::foo::makro as baz;
    //~^ ERROR unresolved import `issue_59764::foo::makro` [E0432]
}

mod renamed_multiple_imports {
    use issue_59764::foo::{baz, makro as foobar};
    //~^ ERROR unresolved import `issue_59764::foo::makro` [E0432]
}

mod lots_of_whitespace {
    use
        issue_59764::{

            foobaz,


            foo::{baz,

                makro as foobar} //~ ERROR unresolved import `issue_59764::foo::makro` [E0432]

        };
}

// Simple case..

use issue_59764::foo::makro;
//~^ ERROR unresolved import `issue_59764::foo::makro` [E0432]

makro!(bar);

fn main() {
    bar();
    //~^ ERROR cannot find function `bar` in this scope [E0425]
}
