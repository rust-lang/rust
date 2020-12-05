// run-pass
trait Stream { type Item; }
impl<'a> Stream for &'a str { type Item = u8; }
fn f<'s>(s: &'s str) -> (&'s str, <&'s str as Stream>::Item) {
    (s, 42)
}

fn main() {
    let fx = f as for<'t> fn(&'t str) -> (&'t str, <&'t str as Stream>::Item);
    assert_eq!(fx("hi"), ("hi", 42));
}
