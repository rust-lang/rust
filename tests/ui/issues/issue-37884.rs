struct RepeatMut<'a, T>(T, &'a ());

impl<'a, T: 'a> Iterator for RepeatMut<'a, T> {

    type Item = &'a mut T;
    fn next(&'a mut self) -> Option<Self::Item>
    //~^ ERROR method not compatible with trait
    //~| lifetime mismatch
    {
        Some(&mut self.0)
    }
}

fn main() {}
