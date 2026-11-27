trait Hello {
    fn example(val: ());
}

struct Test1(i32);

impl Hello for Test1 {
    fn example(&self, input: &i32) {
        //~^ ERROR `&self` declaration in the impl, but not in the trait
        *input = self.0;
        //~^ ERROR cannot assign to `*input`, which is behind a `&` reference
    }
}

fn main() {}
