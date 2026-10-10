//@ compile-flags: -Zmir-opt-level=0 -Zmir-enable-passes=+ScalarReplacementOfAggregates -Aunused

fn main() {
    let t = (10_u8, 20_u16);
}
