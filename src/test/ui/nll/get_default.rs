// Basic test for free regions in the NLL code. This test ought to
// report an error due to a reborrowing constraint. Right now, we get
// a variety of errors from the older, AST-based machinery (notably
// borrowck), and then we get the NLL error at the end.

struct Map {
}

impl Map {
    fn get(&self) -> Option<&String> { None }
    fn set(&mut self, v: String) { }
}

fn ok(map: &mut Map) -> &String {
    loop {
        match map.get() {
            Some(v) => {
                return v;
            }
            None => {
                map.set(String::new()); // Ideally, this would not error.
                //~^ ERROR borrowed as immutable
            }
        }
    }
}

fn err(map: &mut Map) -> &String {
    loop {
        match map.get() {
            Some(v) => {
                map.set(String::new()); // Both AST and MIR error here
                //~^ ERROR borrowed as immutable
                return v;
            }
            None => {
                map.set(String::new()); // Ideally, just AST would error here
                //~^ ERROR borrowed as immutable
            }
        }
    }
}

fn main() { }
