trait T1 : T2 {
//~^ ERROR cycle detected
}

trait T2 : T1 {
}

fn main() { }
