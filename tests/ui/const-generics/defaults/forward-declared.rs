struct Foo<const N: usize = M, const M: usize = 10>;
//~^ ERROR generic parameter defaults cannot reference parameters before they are declared

enum Bar<const N: usize = M, const M: usize = 10> {}
//~^ ERROR generic parameter defaults cannot reference parameters before they are declared

struct Foo2<const N: usize = N>;
//~^ ERROR generic parameter defaults cannot reference parameters before they are declared

enum Bar2<const N: usize = N> {}
//~^ ERROR generic parameter defaults cannot reference parameters before they are declared

fn main() {}
