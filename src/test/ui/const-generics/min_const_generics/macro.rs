// run-pass
struct Example<const N: usize>;

macro_rules! external_macro {
  () => {{
    const X: usize = 1337;
    X
  }}
}

trait Marker<const N: usize> {}
impl<const N: usize> Marker<N> for Example<N> {}

fn make_marker() -> impl Marker<{
    #[macro_export]
    macro_rules! const_macro { () => {{ 3 }} } inline!()
}> {
  Example::<{ const_macro!() }>
}

fn from_marker(_: impl Marker<{
    #[macro_export]
    macro_rules! inline { () => {{ 3 }} } inline!()
}>) {}

fn main() {
  let _ok = Example::<{
    #[macro_export]
    macro_rules! gimme_a_const {
      ($rusty: ident) => {{ let $rusty = 3; *&$rusty }}
    }
    gimme_a_const!(run)
  }>;

  let _ok = Example::<{ external_macro!() }>;

  let _ok: [_; gimme_a_const!(blah)] = [0,0,0];
  let _ok: [[u8; gimme_a_const!(blah)]; gimme_a_const!(blah)];
  let _ok: [u8; gimme_a_const!(blah)];

  let _ok: [u8; {
    #[macro_export]
    macro_rules! const_two { () => {{ 2 }} }
    const_two!()
  }];

  let _ok = [0; {
    #[macro_export]
    macro_rules! const_three { () => {{ 3 }} }
    const_three!()
  }];
  let _ok = [0; const_three!()];

  from_marker(make_marker());
}
