struct send_packet<T: copy> {
  let p: T;
  new(p: T) { self.p = p; }
}


mod pingpong {
    type ping = send_packet<pong>;
    enum pong = send_packet<ping>; //~ ERROR illegal recursive enum type; wrap the inner value in a box to make it representable
}

fn main() {}
