mod foo { pub struct bar; }

fn main() {
    let bar = 5;
    //~^ ERROR mismatched types
    use foo::bar;
}
