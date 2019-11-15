// compile-fail

trait MyIterator {
    fn my_next(&mut self) -> Option<i32>;
}

impl Iterator for dyn MyIterator {
    type Item = i32;

    fn next(&mut self) -> Option<i32> {
        self.my_next()
    }
}

fn use_my_iter(my_iter: &mut dyn MyIterator) {
    let v: Vec<_> = my_iter.map(|i| i*i).collect();
    //~^ ERROR cannot infer an appropriate lifetime
}

struct Wrapper<T>(T);

impl Iterator for Wrapper<&mut dyn MyIterator> {
    type Item = i32;

    fn next(&mut self) -> Option<i32> {
        self.0.my_next()
    }
}

fn use_my_wrapper(wrapper: Wrapper<&mut (dyn MyIterator + '_)>) {
    let v: Vec<_> = wrapper.map(|i| i*i).collect();
    //~^ ERROR cannot infer an appropriate lifetime
}

fn main() {}
