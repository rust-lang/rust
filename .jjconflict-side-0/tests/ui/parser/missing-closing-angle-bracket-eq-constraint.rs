struct Foo<T1, T2> {
  _a : T1,
  _b : T2,
}

fn test1<T>(arg : T) {
  let v : Vec<(u32,_) = vec![];
    //~^ ERROR: expected one of
    //~| ERROR: type annotations needed
}

fn test2<T1, T2>(arg1 : T1, arg2 : T2) {
  let foo : Foo::<T1, T2 = Foo {_a : arg1, _b : arg2};
    //~^ ERROR: expected one of
}

fn test3<'a>(arg : &'a u32) {
  let v : Vec<'a = vec![];
    //~^ ERROR: expected one of
    //~| ERROR: type annotations needed for `Vec<_>`
}

fn main() {}
