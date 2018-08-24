enum bar { t1((), Option<Vec<isize>>), t2, }

// n.b. my change changes this error message, but I think it's right -- tjc
fn foo(t: bar) -> isize { match t { bar::t1(_, Some(x)) => { return x * 3; } _ => { panic!(); } } }
//~^ ERROR binary operation `*` cannot be applied to

fn main() { }
