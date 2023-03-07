// run-pass
// pretty-expanded FIXME #23616

fn foo<'r>() {
  let maybe_value_ref: Option<&'r u8> = None;

  let _ = maybe_value_ref.map(|& ref v| v);
  let _ = maybe_value_ref.map(|& ref v| -> &'r u8 {v});
  let _ = maybe_value_ref.map(|& ref v: &'r u8| -> &'r u8 {v});
  let _ = maybe_value_ref.map(|& ref v: &'r u8| {v});
}

fn main() {
  foo();
}
