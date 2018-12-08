pub   bar<'a>(&self, _s: &'a usize) -> bool { true }
//~^ ERROR missing `fn` for method definition

fn main() {
    bar(2);
}
