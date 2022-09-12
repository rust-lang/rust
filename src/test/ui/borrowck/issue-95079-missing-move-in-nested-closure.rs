fn foo1(s: &str) -> impl Iterator<Item = String> + '_ {
    None.into_iter()
        .flat_map(move |()| s.chars().map(|c| format!("{}{}", c, s)))
        //~^ ERROR captured variable cannot escape `FnMut` closure body
        //~| HELP consider adding 'move' keyword before the nested closure
}

fn foo2(s: &str) -> impl Sized + '_ {
    move |()| s.chars().map(|c| format!("{}{}", c, s))
    //~^ ERROR lifetime may not live long enough
    //~| HELP consider adding 'move' keyword before the nested closure
}

fn main() {}
