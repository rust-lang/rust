// run-pass
// pretty-expanded FIXME #23616

#![allow(noop_method_call)]

struct NoClone;

fn main() {
  let rnc = &NoClone;
  let rsnc = &Some(NoClone);

  let _: &NoClone = rnc.clone();
  let _: &Option<NoClone> = rsnc.clone();
}
