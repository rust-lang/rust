struct S;
impl Foo for S {
    fn parse(s:&str) {
        for c in s.chars() {
            match c {
                '0'..='9' => collect_primary(&c), //~ ERROR cannot find function `collect_primary`
                //~^ HELP you might have meant to call the associated function
                '+' | '-' => println!("We got a sign: {}", c),
                _ => println!("Not a number!")
            }
        }
    }
}
trait Foo {
    fn collect_primary(ch:&char) { }
    fn parse(s:&str);
}
trait Bar {
    fn collect_primary(ch:&char) { }
    fn parse(s:&str) {
        for c in s.chars() {
            match c {
                '0'..='9' => collect_primary(&c), //~ ERROR cannot find function `collect_primary`
                //~^ HELP you might have meant to call the associated function
                '+' | '-' => println!("We got a sign: {}", c),
                _ => println!("Not a number!")
            }
        }
    }
}

fn main() {}
