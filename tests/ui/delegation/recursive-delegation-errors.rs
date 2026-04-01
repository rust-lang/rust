#![feature(fn_delegation)]
#![allow(incomplete_features)]


mod first_mod {
    reuse foo;
    //~^ ERROR failed to resolve delegation callee
}

mod second_mod {
    reuse foo as bar;
    //~^ ERROR encountered a cycle during delegation signature resolution
    reuse bar as foo;
    //~^ ERROR encountered a cycle during delegation signature resolution
}

mod third_mod {
    reuse foo as foo1;
    //~^ ERROR encountered a cycle during delegation signature resolution
    reuse foo1 as foo2;
    //~^ ERROR encountered a cycle during delegation signature resolution
    reuse foo2 as foo3;
    //~^ ERROR encountered a cycle during delegation signature resolution
    reuse foo3 as foo4;
    //~^ ERROR encountered a cycle during delegation signature resolution
    reuse foo4 as foo5;
    //~^ ERROR encountered a cycle during delegation signature resolution
    reuse foo5 as foo;
    //~^ ERROR encountered a cycle during delegation signature resolution
}

mod fourth_mod {
    trait Trait {
        reuse Trait::foo as bar;
        //~^ ERROR encountered a cycle during delegation signature resolution
        reuse Trait::bar as foo;
        //~^ ERROR encountered a cycle during delegation signature resolution
    }
}

mod fifth_mod {
    reuse super::fifth_mod::{bar as foo, foo as bar};
    //~^ ERROR encountered a cycle during delegation signature resolution
    //~| ERROR encountered a cycle during delegation signature resolution

    trait GlobReuse {
        reuse GlobReuse::{foo as bar, bar as goo, goo as foo};
        //~^ ERROR encountered a cycle during delegation signature resolution
        //~| ERROR encountered a cycle during delegation signature resolution
        //~| ERROR encountered a cycle during delegation signature resolution
    }
}

fn main() {}
