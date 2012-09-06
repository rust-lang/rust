// A test case for #2548.

struct foo {
    x: @mut int;


    drop {
        io::println("Goodbye, World!");
        *self.x += 1;
    }
}

fn foo(x: @mut int) -> foo {
    foo { x: x }
}

fn main() {
    let x = @mut 0;

    {
        let mut res = foo(x);
        
        let mut v = ~[mut];
        v <- ~[mut res] + v; //~ ERROR instantiating a type parameter with an incompatible type (needs `copy`, got `owned`, missing `copy`)
        assert (v.len() == 2);
    }

    assert *x == 1;
}
