fn main() {
    |_:  [_; return || {}] | {};
    //~^ ERROR return statement outside of function body

    [(); return || {}];
    //~^ ERROR return statement outside of function body

    [(); return |ice| {}];
    //~^ ERROR return statement outside of function body

    [(); return while let Some(n) = Some(0) {}];
    //~^ ERROR return statement outside of function body
}
