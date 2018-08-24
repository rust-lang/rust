trait t1 : t2 {
//~^ ERROR cycle detected
}

trait t2 : t1 {
}

fn main() { }
