//@ compile-flags: -Znext-solver=globally
//@ check-pass

#![crate_type = "lib"]

// Regression test for trait-system-refactor-initiative#313.

pub trait Ip: Sized {
    type Addr;
}

pub trait DualStackIpExt: Ip {
    type OtherVersion: DualStackIpExt<OtherVersion = Self>;
}

pub trait DatagramDualStackIpExt:
    DualStackIpExt<OtherVersion: DualStackIpExt>
{}

impl<I: DualStackIpExt<OtherVersion: DualStackIpExt>>
    DatagramDualStackIpExt for I
{}

pub trait CoreIpExt:
    DatagramDualStackIpExt
    + DualStackIpExt<OtherVersion: DatagramDualStackIpExt>
{}

impl<I: DatagramDualStackIpExt
    + DualStackIpExt<OtherVersion: DatagramDualStackIpExt>>
    CoreIpExt for I
{}

pub trait TestIpExt: Ip + CoreIpExt {
    type OtherIp: TestIpExt;
    const ADDR: Wrapper<Self::Addr>;
}

pub struct Wrapper<A>(pub A);
pub struct NeighborApi<I>(core::marker::PhantomData<I>);

impl<I: Ip> NeighborApi<I> {
    pub fn resolve_link_addr(&self, _dst: &Wrapper<I::Addr>) {}
}

pub fn repro<I: TestIpExt>(api: NeighborApi<I::OtherIp>) {
    api.resolve_link_addr(&I::OtherIp::ADDR);
}
