struct A {
    x: [(u32, u32); 10]
}

impl A {
    fn iter_values_anon(&self) -> impl Iterator<Item=u32> {
        //~^ ERROR: captures lifetime that does not appear in bounds
        //~| ERROR: captures lifetime that does not appear in bounds
        self.x.iter().map(|a| a.0)
    }
    fn iter_values<'a>(&'a self) -> impl Iterator<Item=u32> {
        //~^ ERROR: captures lifetime that does not appear in bounds
        //~| ERROR: captures lifetime that does not appear in bounds
        self.x.iter().map(|a| a.0)
    }
}

fn main() {}
