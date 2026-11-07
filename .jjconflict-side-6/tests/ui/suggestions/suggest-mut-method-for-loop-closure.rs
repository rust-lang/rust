use std::collections::HashMap;
struct X(usize);
struct Y {
    v: u32,
}

fn main() {
    let _ = || {
        let mut buzz = HashMap::new();
        buzz.insert("a", Y { v: 0 });

        for mut t in buzz.values() {
            //~^ HELP
            //~| SUGGESTION values_mut()
            t.v += 1;
            //~^ ERROR cannot assign
        }
    };
}
