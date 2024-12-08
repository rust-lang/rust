struct AddrVec<H, A> {
    h: H,
    a: A,
}

impl<H> AddrVec<H, DeviceId> {
    //~^ ERROR cannot find type `DeviceId` in this scope
    pub fn device(&self) -> DeviceId {
    //~^ ERROR cannot find type `DeviceId` in this scope
        self.tail()
    }
}

fn main() {}
