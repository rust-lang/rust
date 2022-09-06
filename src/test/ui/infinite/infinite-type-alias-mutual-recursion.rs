type X1 = X2;
//~^ ERROR cycle detected when expanding type alias
type X2 = X3;
//~^ ERROR cycle detected when expanding type alias
type X3 = X1;
//~^ ERROR cycle detected when expanding type alias

fn main() {}
