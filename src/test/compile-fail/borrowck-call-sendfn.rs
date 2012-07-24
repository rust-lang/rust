// xfail-test #2978

fn call(x: @{mut f: fn~()}) {
    x.f(); //~ ERROR foo
    //~^ NOTE bar
}

fn main() {}
