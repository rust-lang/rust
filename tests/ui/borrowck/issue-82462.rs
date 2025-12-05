struct DroppingSlice<'a>(&'a [i32]);

impl Drop for DroppingSlice<'_> {
    fn drop(&mut self) {
        println!("hi from slice");
    }
}

impl DroppingSlice<'_> {
    fn iter(&self) -> std::slice::Iter<'_, i32> {
        self.0.iter()
    }
}

fn main() {
    let mut v = vec![1, 2, 3, 4];
    for x in DroppingSlice(&*v).iter() {
        v.push(*x); //~ERROR
        break;
    }
}
