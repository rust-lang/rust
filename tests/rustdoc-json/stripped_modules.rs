//@ !has "$.index[?(@.name=='no_pub_inner')]"
mod no_pub_inner {
    fn priv_inner() {}
}

//@ !has "$.index[?(@.name=='pub_inner_unreachable')]"
mod pub_inner_unreachable {
    //@ !has "$.index[?(@.name=='pub_inner_1')]"
    pub fn pub_inner_1() {}
}

//@ !has "$.index[?(@.name=='pub_inner_reachable')]"
mod pub_inner_reachable {
    //@ has "$.index[?(@.name=='pub_inner_2')]"
    pub fn pub_inner_2() {}
}

pub use pub_inner_reachable::pub_inner_2;
