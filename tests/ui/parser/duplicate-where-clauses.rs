struct A where (): Sized where (): Sized {}
//~^ ERROR cannot define duplicate `where` clauses on an item

fn b() where (): Sized where (): Sized {}
//~^ ERROR cannot define duplicate `where` clauses on an item

enum C where (): Sized where (): Sized {}
//~^ ERROR cannot define duplicate `where` clauses on an item

struct D where (): Sized, where (): Sized {}
//~^ ERROR cannot define duplicate `where` clauses on an item

fn e() where (): Sized, where (): Sized {}
//~^ ERROR cannot define duplicate `where` clauses on an item

enum F where (): Sized, where (): Sized {}
//~^ ERROR cannot define duplicate `where` clauses on an item

fn main() {}
