// Issue 52126: With respect to variance, the assign-op's like += were
// accidentally lumped together with other binary op's. In both cases
// we were coercing the LHS of the op to the expected supertype.
//
// The problem is that since the LHS of += is modified, we need the
// parameter to be invariant with respect to the overall type, not
// covariant.

use std::collections::HashMap;
use std::ops::AddAssign;

pub fn main() {
    panics();
}

pub struct Counter<'l> {
    map: HashMap<&'l str, usize>,
}

impl<'l> AddAssign for Counter<'l>
{
    fn add_assign(&mut self, rhs: Counter<'l>) {
        rhs.map.into_iter().for_each(|(key, val)| {
            let count = self.map.entry(key).or_insert(0);
            *count += val;
        });
    }
}

/// Often crashes, if not prints invalid strings.
pub fn panics() {
    let mut acc = Counter{map: HashMap::new()};
    for line in vec!["123456789".to_string(), "12345678".to_string()] {
        let v: Vec<&str> = line.split_whitespace().collect();
        //~^ ERROR `line` does not live long enough
        // println!("accumulator before add_assign {:?}", acc.map);
        let mut map = HashMap::new();
        for str_ref in v {
            let e = map.entry(str_ref);
            println!("entry: {:?}", e);
            let count = e.or_insert(0);
            *count += 1;
        }
        let cnt2 = Counter{map};
        acc += cnt2;
        // println!("accumulator after add_assign {:?}", acc.map);
        // line gets dropped here but references are kept in acc.map
    }
}
