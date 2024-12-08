use std::collections::HashMap;



trait Map<K, V>
{
    fn get(&self, k: K) -> V { panic!() }
}

impl<K, V> Map<K, V> for HashMap<K, V> {}

// Test that trait types printed in error msgs include the type arguments.

fn main() {
    let x: Box<HashMap<isize, isize>> = HashMap::new().into();
    let x: Box<dyn Map<isize, isize>> = x;
    let y: Box<dyn Map<usize, isize>> = Box::new(x);
    //~^ ERROR `Box<dyn Map<isize, isize>>: Map<usize, isize>` is not satisfied
}
