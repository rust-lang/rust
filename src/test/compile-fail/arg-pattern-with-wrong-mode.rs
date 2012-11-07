fn f(&&(x, y): (int, int)) {}   //~ ERROR patterns may only be used in arguments with + mode

fn main(){}

