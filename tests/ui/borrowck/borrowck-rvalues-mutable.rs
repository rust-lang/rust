//@ run-pass

struct Counter {
    value: usize
}

impl Counter {
    fn new(v: usize) -> Counter {
        Counter {value: v}
    }

    fn inc<'a>(&'a mut self) -> &'a mut Counter {
        self.value += 1;
        self
    }

    fn get(&self) -> usize {
        self.value
    }

    fn get_and_inc(&mut self) -> usize {
        let v = self.value;
        self.value += 1;
        v
    }
}

pub fn main() {
    let v = Counter::new(22).get_and_inc();
    assert_eq!(v, 22);

    let v = Counter::new(22).inc().inc().get();
    assert_eq!(v, 24);
}
