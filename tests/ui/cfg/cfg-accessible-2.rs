//@ run-pass
//@ edition: 2018
#![feature(cfg_accessible)]

fn main() {
    // accessible
    assert!(cfg!(accessible(::std::boxed::Box)));
    // not accessible because it's internal
    assert!(!cfg!(accessible(::std::vec::RawVec)));
    // not accessible because it's enum variant
    assert!(!cfg!(accessible(::std::net::IpAddr::V4)));
    // not accessible because it's inherent associated constant
    assert!(!cfg!(accessible(::std::time::Duration::ZERO)));
    // not accessible because it's trait associated constant
    assert!(!cfg!(accessible(::std::default::Default::default)));
}
