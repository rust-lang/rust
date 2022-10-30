// Also includes more Self usages per #93796

fn foo(_: Self) {
//~^ ERROR cannot find type `Self`
}

fn foo2() {
    let x: Self;
    //~^ ERROR cannot find type `Self`
}

type Foo<T>
where
    Self: Clone,
//~^ ERROR cannot find type `Self`
= Vec<T>;

const FOO: Self = 0;
//~^ ERROR cannot find type `Self`

const FOO2: u32 = Self::bar();
//~^ ERROR failed to resolve: `Self`

static FOO_S: Self = 0;
//~^ ERROR cannot find type `Self`

static FOO_S2: u32 = Self::bar();
//~^ ERROR failed to resolve: `Self`

fn main() {}
