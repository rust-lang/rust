// run-rustfix

fn _foo(opt: &Option<Box<i32>>) -> String {
    opt.map(|x| x.to_string()).unwrap_or_else(String::new)
    //~^ cannot move out of `*opt` which is behind a shared reference
}

fn main(){}
