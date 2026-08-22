mod a{
    enum EnumAB{
        A,
        B
    }
}
mod b{
    fn enumab_not_in_scope()->enumab{ //~ ERROR
        unimplemented!()
    }
    fn shutdown_not_in_scope()->Shutdown{ //~ ERROR
        unimplemented!()
    }
    fn both_not_in_scope_and_capitalization()->both{ //~ ERROR
        unimplemented!()
    }
    fn shutdown_both_not_in_scope_and_capitalization()->shutdown{ //~ ERROR
        both //~ ERROR
    }
    fn both_missing_std()->std::net::Shutdown{
        net::shutdown::Both //~ ERROR
    }
    fn shutdown_capitalization()->std::net::Shutdown{
        std::net::shutdown::Both //~ ERROR
    }
    fn shutdown_and_both_missing_capitalization()->std::net::Shutdown{
        std::net::shutdown::both //~ ERROR
    }
}
fn main(){}
