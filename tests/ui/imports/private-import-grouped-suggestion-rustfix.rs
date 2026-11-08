//@ run-rustfix
#![allow(unused_imports, dead_code)]

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

mod with_self {
    use crate::b::{self, A};
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

mod single_item {
    use crate::b::{A};
    //~^ ERROR struct import `A` is private [E0603]
}

mod nested_single_in_single {
    use crate::{b::{A}};
    //~^ ERROR struct import `A` is private [E0603]
}

mod pub_use_single {
    pub use crate::b::{A};
    //~^ ERROR struct import `A` is private [E0603]
}

mod with_attribute_single {
    #[allow(unused_imports)]
    use crate::b::{A};
    //~^ ERROR struct import `A` is private [E0603]
}

fn main() {}
