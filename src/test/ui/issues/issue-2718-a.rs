pub struct send_packet<T> {
    p: T
}

mod pingpong {
    use send_packet;
    pub type ping = send_packet<pong>;
    pub struct pong(send_packet<ping>);
    //~^ ERROR recursive type `pingpong::pong` has infinite size
}

fn main() {}
