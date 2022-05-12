// EMIT_MIR enum_prop.main.SingleEnum.diff

fn main() {
  let v = match Some(Box::new(10)) {
    Some(x) => {
      println!("{}", x);
      *x
    },
    _ => 3,
  };
  assert_eq!(v,10);


  let x = match Some(1) {
    ref _y @ Some(_) => 1,
    None => 2,
  };
  assert_eq!(x, 1);

}
