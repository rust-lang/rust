struct Foo {
    x: i32,
}

impl Foo {
    fn this1(&self) -> i32 {
        let this = self;
        let a = 1;
        this.x
    }

    fn this2(&self) -> i32 {
        let a = Foo {
            x: 2
        };
        let this = a;
        this.x
    }

    fn foo(&self) -> i32 {
        this.x
        //~^ ERROR cannot find value `this` in this scope
    }

    fn bar(&self) -> i32 {
        this.foo()
        //~^ ERROR cannot find value `this` in this scope
    }

    fn baz(&self) -> i32 {
        my.bar()
        //~^ ERROR cannot find value `my` in this scope
    }
}

fn main() {
    let this = vec![1, 2, 3];
    let my = vec![1, 2, 3];
    let len = this.len();
    let len = my.len();
}
