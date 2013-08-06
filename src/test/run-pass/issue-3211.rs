pub trait Times {
    fn times(&self, it: &fn());
}

impl Times for uint {
    fn times(&self, it: &fn()) {
        let mut i = *self;
        while i > 0 {
            it();
            i -= 1;
        }
    }
}

pub fn main() {
    let mut x = 0;
    do 4096.times {
        x += 1;
    }
    assert_eq!(x, 4096);
    printfln!("x = %u", x);
}
