// test for issue 84128
// missing suggestion for similar ADT type with diffetent generic paramenter
// on closure ReturnNoExpression

struct Foo<T>(T);

fn main() {
    || {
        if false {
            return Foo(0);
        }

        Foo(())
        //~^ ERROR mismatched types [E0308]
    };
}
