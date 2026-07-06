fn main() {
    let mut samples = Vec::new();
    let packet_buf = Vec::new();
    samples.extend(packet_buf.iter().map(|x| [x, x]));
    //~^ ERROR overflow evaluating whether `&_` is well-formed
    samples.extend(packet_buf.chunks_exact(1).map(|x| [x[0], x[0]]));
}
