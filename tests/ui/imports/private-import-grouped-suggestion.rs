mod a {
    pub struct A;
    pub struct B;
}

mod b {
    use crate::a::{A, B};
    pub struct Z;
}

mod failing_first {
    use crate::b::{A, Z};
    //~^ ERROR struct import `A` is private [E0603]
}

mod failing_last {
    use crate::b::{Z, A};
    //~^ ERROR struct import `A` is private [E0603]
}

mod failing_middle {
    use crate::b::{Z, A, Z as Z2};
    //~^ ERROR struct import `A` is private [E0603]
}

mod single_item {
    use crate::b::{A};
    //~^ ERROR struct import `A` is private [E0603]
}

mod both_failing {
    use crate::b::{A, B};
    //~^ ERROR struct import `A` is private [E0603]
    //~| ERROR struct import `B` is private [E0603]
}

mod multiple_failing_mixed {
    use crate::b::{A, Z, B};
    //~^ ERROR struct import `A` is private [E0603]
    //~| ERROR struct import `B` is private [E0603]
}

mod trailing_comma {
    use crate::b::{Z, A,};
    //~^ ERROR struct import `A` is private [E0603]
}

mod multiline {
    use crate::b::{
        Z,
        A,
        //~^ ERROR struct import `A` is private [E0603]
    };
}

mod rename_inside_group {
    use crate::b::{A as Renamed, Z};
    //~^ ERROR struct import `A` is private [E0603]
}

mod rename_single_item {
    use crate::b::{A as Renamed};
    //~^ ERROR struct import `A` is private [E0603]
}

mod nested_group_multi {
    use crate::{b::{A, Z}, a::B};
    //~^ ERROR struct import `A` is private [E0603]
}

mod nested_group_single {
    use crate::{b::{A}, a::B};
    //~^ ERROR struct import `A` is private [E0603]
}

mod nested_single_in_single {
    use crate::{b::{A}};
    //~^ ERROR struct import `A` is private [E0603]
}

mod pub_use {
    pub use crate::b::{A, Z};
    //~^ ERROR struct import `A` is private [E0603]
}

mod pub_use_single {
    pub use crate::b::{A};
    //~^ ERROR struct import `A` is private [E0603]
}

mod restricted_pub {
    pub(crate) use crate::b::{A, Z};
    //~^ ERROR struct import `A` is private [E0603]
}

mod with_self {
    use crate::b::{self, A};
    //~^ ERROR struct import `A` is private [E0603]
}

mod multibyte_between_items {
    use crate::b::{Z, /* 🦀 日本語 */ A};
    //~^ ERROR struct import `A` is private [E0603]
}

mod non_ascii_sibling_before {
    use crate::b::{Z as 变量, A};
    //~^ ERROR struct import `A` is private [E0603]
}

mod multiline_multibyte_comment {
    use crate::b::{
        Z,
        // 🦀 日本語
        A,
        //~^ ERROR struct import `A` is private [E0603]
    };
}

mod block_comment_adjacent {
    use crate::b::{A /* c */, Z};
    //~^ ERROR struct import `A` is private [E0603]
}

mod with_attribute {
    #[allow(unused_imports)]
    use crate::b::{A, Z};
    //~^ ERROR struct import `A` is private [E0603]
}

mod with_attribute_single {
    #[allow(unused_imports)]
    use crate::b::{A};
    //~^ ERROR struct import `A` is private [E0603]
}

mod inside_function_same_line {
    fn f() { use crate::b::{A, Z}; }
    //~^ ERROR struct import `A` is private [E0603]
}

mod std_path {
    use std::{fs::Path};
    //~^ ERROR struct `Path` is private [E0603]
}

macro_rules! define_use {
    () => {
        use crate::b::{A, Z};
        //~^ ERROR struct import `A` is private [E0603]
    };
}

mod macro_generated {
    define_use!();
}

macro_rules! define_use_single {
    () => {
        use crate::b::{A};
        //~^ ERROR struct import `A` is private [E0603]
    };
}

mod macro_generated_single {
    define_use_single!();
}

mod x {
    pub mod m {
        pub struct Item;
    }
}

mod c {
    use crate::x::m;
    pub struct Other;
}

mod glob_in_group_single {
    use crate::c::{m::*};
    //~^ ERROR module import `m` is private [E0603]
}

mod glob_in_group_multi {
    use crate::c::{m::*, Other};
    //~^ ERROR module import `m` is private [E0603]
}

mod non_glob_control_single {
    use crate::c::{m::Item};
    //~^ ERROR module import `m` is private [E0603]
}

mod non_glob_control_multi {
    use crate::c::{m::Item, Other};
    //~^ ERROR module import `m` is private [E0603]
}

mod unexported {
    mod hidden {
        pub mod m {
            pub struct Item;
        }
    }

    pub mod intermediate {
        use super::hidden::m;
    }
}

mod glob_in_single_group_no_candidate {
    use crate::unexported::intermediate::{m::*};
    //~^ ERROR module import `m` is private [E0603]
}

fn main() {}
