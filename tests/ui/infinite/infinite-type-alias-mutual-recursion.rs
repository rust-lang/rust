type X1 = X2;
//~^ ERROR cycle detected when expanding type alias `X1`
type X2 = X3;
type X3 = X1;

fn main() {}
