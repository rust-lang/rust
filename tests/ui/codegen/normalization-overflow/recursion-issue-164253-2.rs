//@ revisions: current next
//[current]~^ ERROR overflow evaluating the requirement
//@ ignore-compare-mode-next-solver (explicit revisions)
//@[next] compile-flags: -Znext-solver=globally
//@ compile-flags: -Copt-level=0 -Arecursion_depth_exceeding_limit
//@ build-fail
//@ dont-check-compiler-stderr
//@ forbid-output: internal compiler error
//[next]~? ERROR overflow evaluating the requirement

fn main() {
    decode(vec![])
}
fn decode(bitstream: Vec<u64>) {
    let mut bitstream_itr = bitstream.into_iter();
    match decode_packet(&mut bitstream_itr) {
        _ => (),
    }
}
fn decode_packets<I: Iterator<Item = u64>>(itr: &mut I) {
    let res;
    loop {
        match decode_packet(itr) {
            _ => {}
        }
    }
    res
}
fn decode_packet<I: Iterator<Item = u64>>(itr: &mut I) -> Option<()> {
    decode_packets(&mut itr.take(0));
    Some(())
}
