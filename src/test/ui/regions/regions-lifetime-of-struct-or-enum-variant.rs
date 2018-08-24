// This tests verifies that unary structs and enum variants
// are treated as rvalues and their lifetime is not bounded to
// the static scope.

fn id<T>(x: T) -> T { x }

struct Test;

enum MyEnum {
    Variant1
}

fn structLifetime<'a>() -> &'a Test {
  let testValue = &id(Test);
  //~^ ERROR borrowed value does not live long enough
  testValue
}

fn variantLifetime<'a>() -> &'a MyEnum {
  let testValue = &id(MyEnum::Variant1);
  //~^ ERROR borrowed value does not live long enough
  testValue
}


fn main() {}
