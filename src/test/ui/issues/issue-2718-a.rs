pub struct SendPacket<T> {
    p: T
}

mod pingpong {
    use SendPacket;
    pub type Ping = SendPacket<Pong>;
    pub struct Pong(SendPacket<Ping>);
    //~^ ERROR recursive type `pingpong::Pong` has infinite size
}

fn main() {}
