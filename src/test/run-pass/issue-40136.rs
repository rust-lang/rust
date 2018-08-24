macro_rules! m { () => { 0 } }

trait T {
   const C: i32 = m!();
}

struct S;
impl S {
    const C: i32 = m!();
}

fn main() {}
