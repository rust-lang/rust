fn main() {}

macro_rules! expand_to_enum {
    () => {
        enum BadE {}
        //~^ ERROR enum is not supported in `trait`s or `impl`s
        //~| ERROR enum is not supported in `trait`s or `impl`s
        //~| ERROR enum is not supported in `extern` blocks
    };
}

macro_rules! mac_impl {
    ($($i:item)*) => {
        struct S;
        impl S { $($i)* }
    }
}

mac_impl! {
    struct BadS; //~ ERROR struct is not supported in `trait`s or `impl`s
    expand_to_enum!();
}

macro_rules! mac_trait {
    ($($i:item)*) => {
        trait T { $($i)* }
    }
}

mac_trait! {
    struct BadS; //~ ERROR struct is not supported in `trait`s or `impl`s
    expand_to_enum!();
}

macro_rules! mac_extern {
    ($($i:item)*) => {
        extern "C" { $($i)* }
    }
}

mac_extern! {
    struct BadS; //~ ERROR struct is not supported in `extern` blocks
    expand_to_enum!();
}
