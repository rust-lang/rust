pub struct SendPacket<T> {
    p: T
}

mod pingpong {
    use crate::SendPacket;
    pub type Ping = SendPacket<Pong>;
    pub struct Pong(SendPacket<Ping>);
    //~^ ERROR recursive type `Pong` has infinite size
}

fn main() {}
