fn main() {
    match @{foo: true, bar: Some(10), baz: 20} {
      @{foo: true, bar: Some(_), _} => {}
      @{foo: false, bar: None, _} => {}
      @{foo: true, bar: None, _} => {}
      @{foo: false, bar: Some(_), _} => {}
    }
}
