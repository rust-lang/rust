//@ build-pass
// regression test for issue #88307
//@ compile-flags: -C opt-level=s

fn main() {
    if let Some(_val) = Option::<String>::None {}
}
