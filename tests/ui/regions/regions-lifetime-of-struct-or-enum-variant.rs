// This tests verifies that unary structs and enum variants
// are treated as rvalues and their lifetime is not bounded to
// the static scope.

fn id<T>(x: T) -> T {
    x
}

struct Test;

enum MyEnum {
    Variant1,
}

fn struct_lifetime<'a>() -> &'a Test {
    let test_value = &id(Test);
    test_value
    //~^ ERROR cannot return value referencing temporary value
}

fn variant_lifetime<'a>() -> &'a MyEnum {
    let test_value = &id(MyEnum::Variant1);
    test_value
    //~^ ERROR cannot return value referencing temporary value
}

fn main() {}
