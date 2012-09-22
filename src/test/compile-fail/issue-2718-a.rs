struct send_packet<T: Copy> {
  p: T
}


mod pingpong {
    #[legacy_exports];
    type ping = send_packet<pong>;
    enum pong = send_packet<ping>; //~ ERROR illegal recursive enum type; wrap the inner value in a box to make it representable
}

fn main() {}
