//@ run-rustfix

trait Greeter0 {
    fn greet(&self);
}

trait Greeter1 {
    fn greet(&self);
}

type BoxedGreeter = (Box<dyn Greeter0>, Box<dyn Greeter1>);
//~^ HELP to declare that the trait object captures data from argument `self`, you can add a lifetime parameter `'a` in the type alias

struct FixedGreeter<'a>(pub &'a str);

impl Greeter0 for FixedGreeter<'_> {
    fn greet(&self) {
        println!("0 {}", self.0)
    }
}

impl Greeter1 for FixedGreeter<'_> {
    fn greet(&self) {
        println!("1 {}", self.0)
    }
}

struct Greetings(pub Vec<String>);

impl Greetings {
    pub fn get(&self, i: usize) -> BoxedGreeter {
        (Box::new(FixedGreeter(&self.0[i])), Box::new(FixedGreeter(&self.0[i])))
        //~^ ERROR lifetime may not live long enough
    }
}

fn main() {
    let mut g = Greetings {0 : vec!()};
    g.0.push("a".to_string());
    g.0.push("b".to_string());
    g.get(0).0.greet();
    g.get(0).1.greet();
    g.get(1).0.greet();
    g.get(1).1.greet();
}
