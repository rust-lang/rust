// A test case for #2548.

// xfail-test

struct foo {
    x: @mut int;

    new(x: @mut int) { self.x = x; }

    drop {
        io::println("Goodbye, World!");
        *self.x += 1;
    }
}

fn main() {
    let x = @mut 0;

    {
        let mut res = foo(x);
        
        let mut v = ~[mut];
        v <- ~[mut res] + v;
    }

    assert *x == 1;
}
