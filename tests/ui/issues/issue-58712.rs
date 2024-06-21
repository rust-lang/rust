struct AddrVec<H, A> {
    h: H,
    a: A,
}

impl<H> AddrVec<H, DeviceId> {
    //~^ ERROR cannot find type `DeviceId`
    pub fn device(&self) -> DeviceId {
    //~^ ERROR cannot find type `DeviceId`
        self.tail()
    }
}

fn main() {}
